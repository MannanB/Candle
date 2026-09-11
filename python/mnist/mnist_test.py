from mnist.mnist_data import get_data_loader
import candle
import tqdm

(x_train, y_train), (x_test, y_test) = get_data_loader().load_data()

# print(len(x_train), len(y_train), len(x_test), len(y_test))

# print(len(x_train[0]))

BATCH_SIZE = 100
INP_SIZE = 28 * 28
TARGET_NUM = 2
EPOCHS = 5
LR = 0.001

class BinaryClassifier:
    def __init__(self):
        self.l1 = candle.Linear(INP_SIZE, 64)
        self.l2 = candle.Linear(64, 1)

    def forward(self, x):
        x = self.l1.forward(x)
        x = candle.activations.relu(x)
        x = self.l2.forward(x)
        return x

classifier = BinaryClassifier()
optimizer = candle.SGD(lr=LR)

# will need to fix this with candle module or something
# prob just be in python
optimizer.add_layer(classifier.l1)
optimizer.add_layer(classifier.l2)

optimizer.init_grad()

for epoch in range(EPOCHS):
    print("Running", epoch+1)

    pbar = tqdm.tqdm(range(len(x_train) // BATCH_SIZE - 1))
    for ep in pbar :
        train_batch_inp = x_train[ep*BATCH_SIZE : (ep+1) * BATCH_SIZE]
        train_batch_label = []
        for num in y_train[ep*BATCH_SIZE : (ep+1) * BATCH_SIZE]:
            if num == TARGET_NUM: train_batch_label.append(1)
            else: train_batch_label.append(0)

        train_tensor_inp = candle.Tensor(train_batch_inp, requires_grad=False)
        train_tensor_inp = train_tensor_inp.flatten(1, 2).unsqueeze() * (1 / 255.0)
        train_tensor_inp.requires_grad = True

        train_tensor_label = candle.Tensor(train_batch_label, requires_grad=False).unsqueeze().unsqueeze()
        # print(train_tensor_inp.tolist())
        # print()
        # print()
        # print(train_tensor_label.tolist())

        out = classifier.forward(train_tensor_inp)
        # print(classifier.l2.weights)
        # print()
        # print(out.tolist())
        # input()
        loss = candle.losses.mse(out, train_tensor_label)

        pbar.set_description("Loss: " + str(loss.tolist()))

        optimizer.zero_grad()
        loss.backward() # yay
        optimizer.step()

    pbar = tqdm.tqdm(range(len(x_test) // BATCH_SIZE - 1), desc="testing")
    total_correct = 0
    total = 0
    for ep in pbar:
        test_batch_inp = x_test[ep*BATCH_SIZE : (ep+1) * BATCH_SIZE]
        test_batch_label = []
        for num in y_test[ep*BATCH_SIZE : (ep+1) * BATCH_SIZE]:
            if num == TARGET_NUM: test_batch_label.append(1)
            else: test_batch_label.append(0)

        test_tensor_inp = candle.Tensor(test_batch_inp, requires_grad=False)
        test_tensor_inp = test_tensor_inp.flatten(1, 2).unsqueeze() * (1 / 255.0)

        out = classifier.forward(test_tensor_inp)

        preds = out.tolist()
        # print(preds)
        for batch in range(BATCH_SIZE):
            if preds[batch][0][0] > 0.5 and test_batch_label[batch] == 1:
                total_correct += 1
            if preds[batch][0][0] <= 0.5 and test_batch_label[batch] == 0:
                total_correct += 1
            total+=1

    print("Accuracy: " + str(round(total_correct / total * 100)) + "%")