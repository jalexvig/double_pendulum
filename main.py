import functools

import numpy as np
import sympy
from matplotlib import pyplot as plt, animation
from matplotlib.animation import FuncAnimation
from numpy import sin, cos
from scipy.integrate import odeint


def get_expressions():
    """
    Create symoblic expressions for the system.

    Returns:
        Expressions for theta_dd_1 and theta_dd_2.
    """

    s = 'l_1 theta_dd_1 m_1 m_2 l_2 theta_dd_2 theta_1 theta_2 theta_d_2 theta_d_1 g'
    l_1, theta_dd_1, m_1, m_2, l_2, theta_dd_2, theta_1, theta_2, theta_d_2, theta_d_1, g = sympy.symbols(s)
    expr1 = l_1 * theta_dd_1 * (m_1 + m_2) + m_2 * l_2 * (theta_dd_2 * sympy.cos(theta_1 - theta_2) + theta_d_2 ** 2 * sympy.sin(theta_1 - theta_2)) + (m_1 + m_2) * g * sympy.sin(theta_1)
    expr2 = m_2 * l_2 * theta_dd_2 + m_2 * l_1 * (theta_dd_1 * sympy.cos(theta_1 - theta_2) - theta_d_1 ** 2 * sympy.sin(theta_1 - theta_2)) + m_2 * g * sympy.sin(theta_2)
    theta_dd_1_expr = sympy.solveset(expr1, theta_dd_1).args[0]
    theta_dd_2_eval = sympy.solveset(expr2.subs(theta_dd_1, theta_dd_1_expr), theta_dd_2).args[0]
    theta_dd_2_expr = sympy.solveset(expr2, theta_dd_2).args[0]
    theta_dd_1_eval = sympy.solveset(expr1.subs(theta_dd_2, theta_dd_2_expr), theta_dd_1).args[0]

    res1 = sympy.simplify(theta_dd_1_eval)
    res2 = sympy.simplify(theta_dd_2_eval)

    sympy.init_printing()

    print(res1, res2, sep='\n\n')

    return res1, res2


def double_pend(
    y: list,
    t: float,
    m_1: float,
    l_1: float,
    m_2: float,
    l_2: float,
    g: float
) -> list:
    """
    Calculate one step for coupled pendulums.

    Args:
        y: (theta_1, theta_d_1, theta_2, theta_d_2) at this timestep.
        t: Time (unused).
        m_1: Mass 1.
        l_1: Length of pendulum 1.
        m_2: Mass 2.
        l_2: Length of pendulum 2.
        g: Gravity constant.

    Returns:
        Derivative estimates for each element of y.
    """

    theta_1, theta_d_1, theta_2, theta_d_2 = y

    # From get_expressions()
    theta_dd_1 = (g*m_1*sin(theta_1) + g*m_2*sin(theta_1)/2 + g*m_2*sin(theta_1 - 2*theta_2)/2 + l_1*m_2*theta_d_1**2*sin(2*theta_1 - 2*theta_2)/2 + l_2*m_2*theta_d_2**2*sin(theta_1 - theta_2))/(l_1*(-m_1 + m_2*cos(theta_1 - theta_2)**2 - m_2))
    theta_dd_2 = (-g*m_1*sin(theta_2) + g*m_1*sin(2*theta_1 - theta_2) - g*m_2*sin(theta_2) + g*m_2*sin(2*theta_1 - theta_2) + 2*l_1*m_1*theta_d_1**2*sin(theta_1 - theta_2) + 2*l_1*m_2*theta_d_1**2*sin(theta_1 - theta_2) + l_2*m_2*theta_d_2**2*sin(2*theta_1 - 2*theta_2))/(2*l_2*(m_1 - m_2*cos(theta_1 - theta_2)**2 + m_2))

    dydt = [theta_d_1, theta_dd_1, theta_d_2, theta_dd_2]

    return dydt


def plot(
    y0: list,
    t: np.ndarray,
    system_params: dict,
    trajectory: bool = True,
    angles: bool = False,
    show: bool = True,
    save: str = ''
):
    """
    Animate the coupled pendulums.

    Args:
        y0: Initial conditions for (theta_1, theta_d_1, theta_2, theta_d_2).
        t: Times to evaluate equations of motion.
        system_params: Physical parameters of the system.
        trajectory: Animate trajectory.
        angles: Plot angles over time.
        show: Display trajectory animation.
        save: Filepath to save trajectory animation.

    """

    pend_func = functools.partial(double_pend, **system_params)

    theta_1, theta_d_1, theta_2, theta_d_2 = odeint(pend_func, y0, t).T

    if angles:
        theta_1 = theta_1 % np.pi
        theta_2 = theta_2 % np.pi

        plt.plot(t, theta_1)
        plt.plot(t, theta_2)
        plt.show()

    if trajectory:

        l_1, l_2 = system_params['l_1'], system_params['l_2']

        x_1 = l_1 * np.sin(theta_1)
        y_1 = l_1 * np.cos(theta_1)
        x_2 = x_1 + l_2 * np.sin(theta_2)
        y_2 = y_1 + l_2 * np.cos(theta_2)

        fig, ax = plt.subplots()

        ln, = ax.plot([], [], 'o-', lw=2)

        def init():
            bound = l_1 + l_2
            ax.set_xlim(-bound, bound)
            ax.set_ylim(-bound, bound)
            ln.set_data([], [])
            return ln,

        def update(vals):
            ln.set_data(vals[:3], vals[3:])
            return ln,

        frames = np.stack([np.zeros_like(x_1), x_1, x_2, np.zeros_like(y_1), -y_1, -y_2], axis=-1)

        ani = FuncAnimation(fig, update, init_func=init, frames=frames, blit=True, interval=25, repeat=False)

        if save:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=40, metadata=dict(artist='Me'), bitrate=1800)

            ani.save(save, writer=writer)

        if show:
            plt.show()


if __name__ == '__main__':

    system_params = {
        'm_1': 1,
        'l_1': 1,
        'm_2': 1,
        'l_2': 1,
        'g': 9.81
    }

    # Initial conditions for (theta_1, theta_d_1, theta_2, theta_d_2)
    y0 = [np.pi / 2, 0, np.pi, 0]

    t = np.linspace(0, 20, 1000)

    plot(y0, t, system_params)
