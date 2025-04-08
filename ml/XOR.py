import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def relu(x: np.ndarray) -> np.ndarray:
    return x * (x > 0)

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, 1, alpha)

def post_processing(predictions: np.ndarray) -> np.ndarray:
    return np.where(predictions < 0.5, 0, 1)

def display_results(inputs: np.ndarray, predictions: np.ndarray) -> None:
    processed_predictions = post_processing(predictions)
    print("Input (A, B) | Predicted Y")
    print("---------------------------")
    for i in range(inputs.shape[1]):
        print(f"   {inputs[0, i]}, {inputs[1, i]}    |     {processed_predictions[0, i]}")

def initialize_parameters() -> dict[str, np.ndarray]:
    parameters = {
        "W1": np.random.randn(2, 2),  # 가중치 | INPUT(2 units) -> Hidden Layer(2 units)
        "b1": np.zeros((2, 1)),       # 편향 | Hidden Layer(2 units)
        "W2": np.random.randn(1, 2),  # 가중치 | Hidden Layer(2 units) -> Output(1 unit)
        "b2": np.zeros((1, 1))        # 편향 | Output(1 unit)
    }
    return parameters

def compute_loss(Y: np.ndarray, Y_hat: np.ndarray) -> float:
    # BCE (Binary Cross Entropy)
    m = Y.shape[1]
    loss = -np.sum(Y * np.log(Y_hat + 1e-8) + (1 - Y) * np.log(1 - Y_hat + 1e-8)) / m
    return loss

def forward_propagation(
    X: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = leaky_relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    return A1, A2

def backward_propagation(
    parameters: dict[str, np.ndarray],
    A1: np.ndarray,
    A2: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
) -> dict[str, np.ndarray]:
    m = X.shape[1]
    W2 = parameters["W2"]

    dZ2 = A2 - Y  # BCE 손실 함수의 gradient
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * leaky_relu_derivative(A1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return gradients

def update_parameters(
    parameters: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    learning_rate: float = 0.1,
) -> dict[str, np.ndarray]:
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]

    return parameters

# XOR 문제에 대한 입력과 출력 정의
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # (2,4)
outputs = np.array([[0, 1, 1, 0]])  # (1,4)

# 파라미터 초기화와 순전파 실행
parameters = initialize_parameters()
predicted_outputs = forward_propagation(inputs, parameters)[1]

# 예측 결과 출력
display_results(inputs, predicted_outputs)

for i in range(1000000):
    A1, A2 = forward_propagation(inputs, parameters)
    grads = backward_propagation(parameters, A1, A2, inputs, outputs)
    parameters = update_parameters(parameters, grads, learning_rate=0.1)
    loss = compute_loss(outputs, A2)

    if i > 0 and i % 10000 == 0:
        print(f"{i=}, {loss=}")

predicted_outputs = forward_propagation(inputs, parameters)[1]
print(predicted_outputs)
display_results(inputs, predicted_outputs)

def forward_propagation_for_debuging(X, parameters) -> tuple[np.ndarray, np.ndarray]:
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    print(f"{X=}")
    print(f"{W1=}")
    print(f"{b1=}")
    print(f"{W2=}")
    print(f"{b2=}")

    Z1 = np.dot(W1, X) + b1
    print(f"{Z1=}")
    A1 = leaky_relu(Z1)
    print(f"{A1=}")

    Z2 = np.dot(W2, A1) + b2
    print(f"{Z2=}")
    A2 = sigmoid(Z2)
    print(f"{A2=}")

    return A1, A2

predicted_outputs = forward_propagation_for_debuging(inputs, parameters)[1]
print(predicted_outputs)