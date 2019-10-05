from sklearn import datasets
digits = datasets.load_digits()

y = digits.target
x = digits.images.reshape((len(digits.images), -1))
x.shape

x_train = x[:1000]
y_train = y[:1000]
x_test = x[1000:]
y_test = y[1000:]

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(15,), activation='logistic', alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.1, verbose=True)

mlp.fit(x_train,y_train)

predictions = mlp.predict(x_test)
predictions[:50]

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))



from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
initial_type = [('input', FloatTensorType([1, 64]))]
onx = convert_sklearn(mlp, initial_types=initial_type)

print("ok")
with open("digits.onnx", "wb") as f:
    f.write(onx.SerializeToString())

from winmltools.utils import save_text
save_text(onx, "digits.json")

print(x[1])
print(mlp.predict([x[1]]))
