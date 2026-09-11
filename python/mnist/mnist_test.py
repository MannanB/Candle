from mnist.mnist_data import get_data_loader
import candle
import tqdm

(x_train, y_train), (x_test, y_test) = get_data_loader().load_data()

print(len(x_train), len(y_train), len(x_test), len(y_test))

print(len(x_train[0]), y_train[0])

BATCH_SIZE = 50
INP_SIZE = 28 * 28
TARGET_NUM = 2
EPOCHS = 1

class BinaryClassifier:
    def __init__(self):
        self.l1 = candle.Linear(INP_SIZE, 16)
        self.l2 = candle.Linear(16, 1)

    def forward(self, x):
        x = self.l1.forward(x)
        x = candle.activations.relu(x)
        x = self.l2.forward(x)
        return x

classifier = BinaryClassifier()

for epoch in range(EPOCHS):
    print("Running", epoch+1)

    for ep in tqdm.tqdm(range(x_train // BATCH_SIZE - 1)):
        train_batch_inp = x_train[ep*BATCH_SIZE : ep * (BATCH_SIZE+1)]
        train_batch_label = [1 for num in y_train[ep*BATCH_SIZE : ep * (BATCH_SIZE+1)] if num==TARGET_NUM else 0]

        train_tensor_inp = candle.Tensor(train_batch_inp, requires_grad=False)
        train_tensor_inp = train_tensor_inp.flatten(1, 2)
        train_tensor_inp.requires_grad = True

        train_tensor_label = candle.Tensor(train_batch_label, requires_grad=False)

        out = classifier.forward(train_batch_inp)
        loss = candle.losses.mse(out, train_tensor_label)

        loss.backward() # yay