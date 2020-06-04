import matplotlib.pyplot as plt
import numpy as np


def riskFunc(x, thr, reward_amount):
    return [ reward_amount * 1 / each if each < thr else 0 for each in x]

def riskFunc2(x, thr, reward_amount):
    return [reward_amount * ( 1 / each - 1 / thr)**2 if each < thr else 0 for each in x]

def riskFunc3(x, thr, reward_amount, k):
    return [reward_amount - k * each if each < thr else 0 for each in x]

x = np.arange(1, 41, 1)
for index, thr in enumerate([10, 20, 30, 40]):
    reward_amount = 10
    # y = riskFunc(x, thr, reward_amount)
    # y = riskFunc2(x, thr, reward_amount)
    y = riskFunc3(x, thr, reward_amount, 0.2)
    plt.subplot(2, 2, index + 1)
    plt.title("Potential Risk Function [thr = {}]".format(thr), fontsize=15)


    plt.plot(x[:thr - 1], y[:thr - 1], "bo-", lw = 2, label = "Threshold = {}".format(thr))
    plt.plot(x[thr:], y[thr:], "bo-", lw = 1.5)
    # plt.plot(x, y, "ro-", lw=2, label="Threshold = {}".format(thr))
    plt.plot([0, 40], [reward_amount, reward_amount], "k--", alpha = 0.5)

    plt.scatter([11], y[11],  color="red", s= 300, marker="*")

    plt.ylim(0, 11)
    plt.xlim(0, 40)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)

    plt.ylabel("Potential Risk", fontsize = 15)
    plt.xlabel("Ghost Distance $D_G$", fontsize = 15)

plt.tight_layout()
plt.show()