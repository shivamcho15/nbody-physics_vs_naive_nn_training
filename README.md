# nbody-physics_vs_naive_nn_training
"When you train a neural network to roll out a chaotic 3-body gravitational system, **how much does baking in physics structure actually help?**: Compared against a "predict the next state" MLP, a physics-aware network that only predicts accelerations and lets an integrator handle the rest is **~28× more accurate at short time scales**.
