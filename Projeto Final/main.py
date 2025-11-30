"""This tutorial shows how to train an MATD3 agent on the simple speaker listener multi-particle environment.

Authors: Michael (https://github.com/mikepratt1), Nickua (https://github.com/nicku-a)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd # Importar pandas para el promedio móvil
from pettingzoo.mpe import simple_speaker_listener_v4
from mpe2 import simple_speaker_listener_v4

# Cambiar la importación del algoritmo
from agilerl.algorithms import MADDPG # <--- NUEVO ALGORITMO
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import (
    create_population,
    default_progress_bar,
    make_multi_agent_vect_envs,
)

# Función para suavizar las puntuaciones con un promedio móvil
def smooth_scores(scores, window=50):
    """Calcula el promedio móvil de una lista de puntuaciones."""
    # Convertir a Series de pandas para usar la función rolling().mean()
    scores_series = pd.Series(scores)
    # Calcular el promedio móvil, rellenando los primeros valores con los valores originales
    # hasta que haya suficientes puntos para el tamaño de la ventana.
    smoothed = scores_series.rolling(window=window, min_periods=1).mean().tolist()
    return smoothed

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== AgileRL Online Multi-Agent Demo =====")

    # Define the network configuration
    # Aumentada capacidad de la red para una mejor aproximación de funciones
    NET_CONFIG = {
        "latent_dim": 128,  # Aumentado de 64 para mejor aprendizaje de representación
        "encoder_config": {
            "hidden_size": [128, 128],  # Red del actor más profunda para coordinación compleja
        },
        "head_config": {
            "hidden_size": [128, 128],  # Red del crítico más profunda para una mejor estimación de valor
        },
    }

    # Define the initial hyperparameters
    # Optimizado para un mejor rendimiento en el entorno Speaker-Listener
    INIT_HP = {
        "POPULATION_SIZE": 4,
        "ALGO": "MADDPG",  # <--- NUEVO ALGORITMO
        "BATCH_SIZE": 256,  # Aumentado de 128 para estimaciones de gradiente más estables
        "EXPL_NOISE": 0.1,  # Reducido para MADDPG/TD3-like
        "LR_ACTOR": 0.0003,  # Tasa de aprendizaje del actor
        "LR_CRITIC": 0.001,  # Tasa de aprendizaje del crítico
        "GAMMA": 0.99,  # Aumentado de 0.95 para valorar recompensas a largo plazo (alcanzar el objetivo)
        "MEMORY_SIZE": 100000,  # Tamaño máximo del buffer de memoria
        "LEARN_STEP": 50,  # Reducido de 100 para un aprendizaje más frecuente
        "TAU": 0.005,  # Reducido de 0.01 para actualizaciones de destino más lentas y estables
        # Parámetros específicos de MATD3 eliminados o ajustados para MADDPG
    }

    num_envs = 8

    def make_env():
        return simple_speaker_listener_v4.parallel_env(continuous_actions=True)

    env = make_multi_agent_vect_envs(env=make_env, num_envs=num_envs)

    # Configure the multi-agent algo input arguments
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["AGENT_IDS"] = env.agents

    # Mutation config for RL hyperparameters
    hp_config = HyperparameterConfig(
        lr_actor=RLParameter(min=1e-4, max=1e-2),
        lr_critic=RLParameter(min=1e-4, max=1e-2),
        batch_size=RLParameter(min=8, max=512, dtype=int),
        learn_step=RLParameter(
            min=20, max=200, dtype=int, grow_factor=1.5, shrink_factor=0.75
        ),
    )

    # Create a population ready for evolutionary hyper-parameter optimisation
    # La población ahora se crea con MADDPG
    pop: list[MADDPG] = create_population(
        INIT_HP["ALGO"],
        observation_spaces,
        action_spaces,
        NET_CONFIG,
        INIT_HP,
        hp_config=hp_config,
        population_size=INIT_HP["POPULATION_SIZE"],
        num_envs=num_envs,
        device=device,
    )

    # Configure the multi-agent replay buffer
    field_names = ["obs", "action", "reward", "next_obs", "done"]
    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        agent_ids=INIT_HP["AGENT_IDS"],
        device=device,
    )

    # Instantiate a tournament selection object (used for HPO)
    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )

    # Instantiate a mutations object (used for HPO)
    mutations = Mutations(
        no_mutation=0.2,  # Probability of no mutation
        architecture=0.2,  # Probability of architecture mutation
        new_layer_prob=0.2,  # Probability of new layer mutation
        parameters=0.2,  # Probability of parameter mutation
        activation=0,  # Probability of activation function mutation
        rl_hp=0.2,  # Probability of RL hyperparameter mutation
        mutation_sd=0.1,  # Mutation strength
        rand_seed=1,
        device=device,
    )

    # Define training loop parameters
    max_steps = 1_000_000  # Aumentado de 2M para mejor convergencia
    learning_delay = 0  # Pasos antes de comenzar el aprendizaje
    evo_steps = 10_000  # Frecuencia de la evolución
    eval_steps = None  # Pasos de evaluación por episodio - ir hasta el final
    eval_loop = 1  # Número de episodios de evaluación
    elite = pop[0]  # Asignar un agente "élite" placeholder
    total_steps = 0
    
    # Lista para almacenar pontuações médias para plotagem
    training_scores_history = []

    # TRAINING LOOP
    print("Training...")
    pbar = default_progress_bar(max_steps)
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []
        for agent in pop:  # Loop through population
            agent.set_training_mode(True)
            obs, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0
            for idx_step in range(evo_steps // num_envs):
                # La función get_action no necesita noise específico de OU si el algoritmo no lo usa (MADDPG solo usa EXPL_NOISE)
                action, raw_action = agent.get_action(
                    obs=obs, infos=info
                )  # Predict action
                next_obs, reward, termination, truncation, info = env.step(
                    action
                )  # Act in environment

                scores += np.sum(np.array(list(reward.values())).transpose(), axis=-1)
                total_steps += num_envs
                steps += num_envs

                # Save experiences to replay buffer
                memory.save_to_memory(
                    obs,
                    raw_action,
                    reward,
                    next_obs,
                    termination,
                    is_vectorised=True,
                )

                # Learn according to learning frequency
                # Handle learn steps > num_envs
                if agent.learn_step > num_envs:
                    learn_step = agent.learn_step // num_envs
                    if (
                        idx_step % learn_step == 0
                        and len(memory) >= agent.batch_size
                        and memory.counter > learning_delay
                    ):  
                        experiences = memory.sample(
                            agent.batch_size
                        )  # Sample replay buffer
                        agent.learn(
                            experiences
                        )  # Learn according to agent's RL algorithm

                # Handle num_envs > learn step; learn multiple times per step in env
                elif (
                    len(memory) >= agent.batch_size and memory.counter > learning_delay
                ):
                    for _ in range(num_envs // agent.learn_step):
                        experiences = memory.sample(
                            agent.batch_size
                        )  # Sample replay buffer
                        agent.learn(
                            experiences
                        )  # Learn according to agent's RL algorithm

                obs = next_obs

                # Calculate scores and reset noise for finished episodes
                reset_noise_indices = []
                term_array = np.array(list(termination.values())).transpose()
                trunc_array = np.array(list(truncation.values())).transpose()
                for idx, (d, t) in enumerate(zip(term_array, trunc_array)):
                    if np.any(d) or np.any(t):
                        completed_episode_scores.append(scores[idx])
                        agent.scores.append(scores[idx])
                        scores[idx] = 0
                        reset_noise_indices.append(idx)

                # Reset noise for the agent. If MADDPG, only the exploration noise is reset.
                agent.reset_action_noise(reset_noise_indices)

            pbar.update(evo_steps // len(pop))

            agent.steps[-1] += steps
            pop_episode_scores.append(completed_episode_scores)

        # Evaluate population
        fitnesses = [
            agent.test(
                env,
                max_steps=eval_steps,
                loop=eval_loop,
            )
            for agent in pop
        ]
        mean_scores = [
            (
                np.mean(episode_scores)
                if len(episode_scores) > 0
                else 0
            )
            for episode_scores in pop_episode_scores
        ]
        
        # Salvar pontuação média da população para plotagem
        population_mean_score = np.mean([score for score in mean_scores if isinstance(score, (int, float))])
        training_scores_history.append(population_mean_score)

        mean_scores_display = [
            (
                score if isinstance(score, (int, float))
                else "0 completed episodes"
            )
            for score in mean_scores
        ]

        pbar.write(
            f"--- Global steps {total_steps} ---\n"
            f"Steps {[agent.steps[-1] for agent in pop]}\n"
            f"Scores: {mean_scores_display}\n"
            f"Fitnesses: {['%.2f' % fitness for fitness in fitnesses]}\n"
            f"5 fitness avgs: {['%.2f' % np.mean(agent.fitness[-5:]) for agent in pop]}\n"
            f"Mutations: {[agent.mut for agent in pop]}"
        )

        # Tournament selection and population mutation
        elite, pop = tournament.select(pop)
        pop = mutations.mutation(pop)

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

    # Save the trained algorithm
    path = "./models/MADDPG" # Cambiar el nombre de la carpeta para MADDPG
    filename = "MADDPG_trained_agent.pt"
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    elite.save_checkpoint(save_path)
    
    # --- PLOTEAR CON PROMEDIO MÓVIL ---
    # Parámetro de la ventana para el promedio móvil
    SMOOTHING_WINDOW = 50 
    smoothed_scores = smooth_scores(training_scores_history, window=SMOOTHING_WINDOW)

    plt.figure(figsize=(12, 6))
    # Plotear la línea suavizada
    plt.plot(smoothed_scores, linewidth=2, label=f'Promedio Móvil (Ventana={SMOOTHING_WINDOW})') 
    # Opcional: plotear los puntos brutos en gris claro para contexto
    plt.plot(training_scores_history, alpha=0.3, label='Puntuación Media Bruta')

    plt.axhline(y=-60, color='r', linestyle='--', label='Objetivo de Rendimiento (-60)') # Añadir la línea de rendimiento objetivo

    plt.title('Evolución de las Puntuaciones Medias (MADDPG) Durante el Entrenamiento', fontsize=14)
    plt.xlabel('Iteraciones de Evolución', fontsize=12)
    plt.ylabel('Pontuação Média de la População', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    
    # Salvar el gráfico
    plot_path = os.path.join(path, "training_scores_evolution_smoothed.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico de evolución de las puntuaciones suavizado salvado en: {plot_path}")
    
    # Salvar datos de las puntuaciones en archivo numpy
    scores_data_path = os.path.join(path, "training_scores_history.npy")
    np.save(scores_data_path, np.array(training_scores_history))
    print(f"Datos de las puntuaciones salvados en: {scores_data_path}")
    
    plt.show()

    pbar.close()
    env.close()