from multiprocessing import Queue
import matplotlib.pyplot as plt
import numpy as np
import time
import signal
import math

def plot_training(q1: Queue, q2: Queue, update_interval=0.01):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    plt.ion()
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle("Training...")
    fig.tight_layout(pad=3.0)

    graph11, = axs[0, 0].plot([], [], color="g", alpha=0.5)
    graph12 = axs[0, 0].axhline(0, color="r", linestyle="--")
    graph13, = axs[0, 0].plot([], [], color="b")

    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].grid(axis = 'y')
    axs[0, 0].margins(0)
    axs[0, 0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    
    q_value_graphs = []

    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Mean Q-Values")
    axs[1, 0].grid(axis = 'y')
    axs[1, 0].margins(0)
    axs[1, 0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    graph21, = axs[0, 1].plot([], [], color="g", alpha=0.5)
    graph22, = axs[0, 1].plot([], [], color="b")

    axs[0, 1].set_xlabel("Step")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].grid(axis = 'y')
    axs[0, 1].margins(0)
    axs[0, 1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    graph31, = axs[1, 1].plot([], [], color="g")

    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Epsilon")
    axs[1, 1].grid(axis = 'y')
    axs[1, 1].margins(0)
    axs[1, 1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    scores = []
    best_avg_reward = -float("inf")
    q_values_buffer = []

    epsilons = []
    losses = []
    durations = []

    try:
        while plt.fignum_exists(fig.number):
            while not q1.empty():
                scores, best_avg_reward, q_values_mean, epsilon, step_duration  = q1.get_nowait()
                q_values_buffer.append(q_values_mean)
                epsilons.append(epsilon)
                durations.append(step_duration)

            graph11.set_data([i for i in range(0, len(scores))], scores)
            graph12.set_ydata([best_avg_reward, best_avg_reward])
            graph13.set_data([np.float32(0)] + [i for i in range(100, len(scores), math.ceil(len(scores) / 20))] + [len(scores)], [np.float32(scores[0])] + [np.mean(scores[i - 100:i]) for i in range(100, len(scores), math.ceil(len(scores) / 20))] + [np.mean(scores[-100:])])

            axs[0, 0].relim()
            axs[0, 0].autoscale_view()

            if len(q_value_graphs) <= 0:
                q_value_graphs = [axs[1, 0].plot([], [], alpha=0.5)[0] for _ in range(len(q_values_buffer[0]))]

            for graph, q_values in zip(q_value_graphs, list(zip(*q_values_buffer))):
                graph.set_data([[i for i in range(len(q_values))], q_values])

            axs[1, 0].relim()
            axs[1, 0].autoscale_view()

            while not q2.empty():
                loss = q2.get_nowait()
                losses.append(loss) if loss is not None else None

            graph21.set_data([i for i in range(0, len(losses))], losses)
            graph22.set_data([i for i in range(0, len(losses), 100)] + [len(losses)], [np.mean(losses[i - 100:i]) for i in range(0, len(losses), 100)] + [np.mean(losses[-100:])])

            axs[0, 1].relim()
            axs[0, 1].autoscale_view()

            graph31.set_data([i for i in range(0, len(epsilons))], epsilons)

            axs[1, 1].relim()
            axs[1, 1].autoscale_view()

            fig.canvas.draw_idle()
            fig.suptitle(f"Training... (Eps: {len(scores)}; Steps: {len(losses)}; Steps/s: {np.mean(durations):.0f})")
            plt.pause(0.001)

            time.sleep(update_interval)
    except KeyboardInterrupt:
        pass
