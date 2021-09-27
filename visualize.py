import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def compare_fasttext(model="tcfasttext", ac="_p"):
    prefix = "modelfiles/" + model

    with open(prefix + ac + "s/history.txt", "r", encoding="utf-8") as fr:
        history_s = fr.read()
        history_s = eval(history_s)

    with open(prefix + ac + "l/history.txt", "r", encoding="utf-8") as fr:
        history_l = fr.read()
        history_l = eval(history_l)

    gs = gridspec.GridSpec(2, 6)

    plt.subplot(gs[0, 1:3])
    plt.plot(history_s["val_loss"])
    plt.plot(history_l["val_loss"])
    plt.title('val_loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['original', 'aeda'], loc='best', prop={'size': 4})

    plt.subplot(gs[0, 3:-1])
    plt.plot(history_s["val_acc"])
    plt.plot(history_l["val_acc"])
    plt.title('val_acc')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['original', 'aeda'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, :2])
    plt.plot(history_s["val_precision"])
    plt.plot(history_l["val_precision"])
    plt.title('val_precision')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['original', 'aeda'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, 2:4])
    plt.plot(history_s["val_recall"])
    plt.plot(history_l["val_recall"])
    plt.title('val_recall')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['original', 'aeda'], loc='best', prop={'size': 4})

    plt.subplot(gs[1, 4:])
    plt.plot(history_s["val_F1"])
    plt.plot(history_l["val_F1"])
    plt.title('val_F1')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['original', 'aeda'], loc='best', prop={'size': 4})

    plt.suptitle("Model Metrics")

    plt.tight_layout()
    plt.savefig("compare_" + model + ac + ".jpg", dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    compare_fasttext(model="tcfasttext", ac="_p")
