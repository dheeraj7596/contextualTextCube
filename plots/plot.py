import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 20 news fine
    # plt_micro = [0.58, 0.63, 0.64, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65]
    # plt_macro = [0.57, 0.62, 0.63, 0.63, 0.64, 0.64, 0.64, 0.64, 0.64]

    # 20 new coarse
    # plt_micro = [0.58, 0.59, 0.60, 0.65, 0.61, 0.61, 0.62, 0.62, 0.62]
    # plt_macro = [0.53, 0.56, 0.56, 0.56, 0.56, 0.57, 0.57, 0.57, 0.57]

    # nyt fine
    # plt_micro = [0.76, 0.88, 0.90, 0.90, 0.90, 0.91, 0.91, 0.91, 0.91]
    # plt_macro = [0.66, 0.77, 0.78, 0.79, 0.79, 0.78, 0.77, 0.77, 0.79]

    plt.figure()
    plt_x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt_micro = [0.92, 0.93, 0.93, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
    plt_macro = [0.85, 0.85, 0.87, 0.88, 0.89, 0.89, 0.89, 0.89, 0.89]
    plt.plot(plt_x, plt_micro, label="Micro F1")
    plt.plot(plt_x, plt_macro, label="Macro F1")
    plt.xlabel("Number of iterations", fontsize=22)
    plt.ylabel("F1 score", fontsize=25)
    plt.legend(prop={'size': 25})
    plt.savefig('./nyt_coarse.png')
    # plt.show()
