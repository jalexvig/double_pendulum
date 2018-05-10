# Double Pendulum

This code implements equations of motion for a coupled system of pendulums.

See [this derivation and explanation](http://jalexvig.github.io/blog/double-pendulum/) for more details.

# To run

Make sure you have python3 installed.

    git clone https://github.com/jalexvig/double_pendulum
    cd double_pendulum

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    
    python main.py 
    
To change the parameters of the system (e.g. pendulum masses/lengths), open `main.py` and modify the dictionary `system_params`:

    system_params = {
        'm_1': 1,
        'l_1': 1,
        'm_2': 1,
        'l_2': 1,
        'g': 9.81
    }
