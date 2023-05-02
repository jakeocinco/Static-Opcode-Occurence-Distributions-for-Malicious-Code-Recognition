import numpy as np
from classifiers.MultiLayerClassifier import MultiLayerClassifier

import pickle
from sklearn.neural_network import MLPClassifier

TEST_SIZE = 200

ITER = 3
for BIN in [25, 100]:

    X = np.load(open(f"./X_2000_7_{BIN}_two_sided_log.npy", "rb"))
    y = np.load(open(f"./y_2000_7_{BIN}_two_sided_log.npy", "rb"))

    model_torch = MultiLayerClassifier(
        input=X.shape[1],
        hidden_layers=[100, 200, 50], #[256, 512, 256, 96, 32, 1],
        drop_out=0.07
    )
    model_sklearn = MLPClassifier(
        hidden_layer_sizes=(100, 200, 50),
        random_state=1,
        max_iter=500
    )

    X_test = X[-TEST_SIZE:]
    y_test = y[-TEST_SIZE:]

    X = X[:-TEST_SIZE]
    y = y[:-TEST_SIZE]

    print(f"Train Size - {X.shape}/{y.shape}")
    print(f"Test Size - {X_test.shape}/{y_test.shape}")


    model_torch.fit(
        X, y,
        X_test=X_test,
        y_test=y_test,
        training_stages=[(0.0001, 500)]
    ) #, (0.0001, 200)])

    predctions = model_torch.predict(X_test)
    print(f"Torch: {sum(predctions == y_test) / TEST_SIZE}")


    model_sklearn.fit(X, y)

    predctions = model_sklearn.predict(X_test)
    print(f"Sklearn: {sum(predctions == y_test) / TEST_SIZE}")

    pickle.dump(model_torch, open(f'./models/torch_{BIN}_two_sided_log.pickle', "+wb"))
    pickle.dump(model_sklearn, open(f'./models/sklearn_{BIN}_two_sided_log.pickle', "+wb"))
