import matplotlib.pyplot as plt
from libemg.datasets import OneSubjectMyoDataset
from libemg.emg_predictor import EMGClassifier
from libemg.feature_extractor import FeatureExtractor
from libemg.offline_metrics import OfflineMetrics

WINDOW_SIZE = 20
WINDOW_INCREMENT = 10
FEATURE_SET = "HTD"

def main():
    dataset = OneSubjectMyoDataset()
    data = dataset.prepare_data()

    train_data = data['Train']
    test_data  = data['Test']

    train_windows, train_meta = train_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
    test_windows, test_meta   = test_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)

    fe = FeatureExtractor()
    om = OfflineMetrics()

    classifiers = ["LDA","SVM","KNN","RF","QDA"]

    training_features = fe.extract_feature_group(FEATURE_SET, train_windows)
    dataset_dict = {
        "training_features": training_features,
        "training_labels": train_meta["classes"]
    }
    test_features = fe.extract_feature_group(FEATURE_SET, test_windows)

    cas, aers, inss = [], [], []

    best_model = None
    best_accuracy = 0.0
    best_clf_name = None

    for clf_name in classifiers:
        print(f"Training and evaluating {clf_name}...")
        model = EMGClassifier(clf_name)

        model.fit(dataset_dict.copy())

        preds, probs = model.run(test_features)

        metrics = om.extract_common_metrics(test_meta["classes"], preds, null_label=2)
        ca = metrics["CA"] * 100
        aer = metrics["AER"] * 100
        ins = metrics["INS"] * 100

        cas.append(ca)
        aers.append(aer)
        inss.append(ins)

        print(f"{clf_name} => CA: {ca:.2f}%, AER: {aer:.2f}%, INS: {ins:.2f}%")

        if ca > best_accuracy:
            best_accuracy = ca
            best_model = model
            best_clf_name = clf_name

    fig, axs = plt.subplots(3, figsize=(7, 9))
    axs[0].bar(classifiers, cas)
    axs[0].set_title("Classification Accuracy (CA)")
    axs[0].set_ylabel("Percent (%)")

    axs[1].bar(classifiers, aers)
    axs[1].set_title("Active Error (AER)")
    axs[1].set_ylabel("Percent (%)")

    axs[2].bar(classifiers, inss)
    axs[2].set_title("Instability (INS)")
    axs[2].set_ylabel("Percent (%)")
    plt.tight_layout()
    plt.show()

    fe.visualize_feature_space(
        dataset_dict['training_features'],
        projection="PCA",
        classes=train_meta['classes'],
        test_feature_dic=test_features,
        t_classes=test_meta['classes']
    )

    if best_model is not None:
        best_model_filename = "my_pretrained_classifier.pkl"
        best_model.save(best_model_filename)
        print(f"Best model: {best_clf_name} with CA={best_accuracy:.2f}%.")
        print(f"Saved best model to '{best_model_filename}'")
    else:
        print("No best model found (no classifiers trained).")

    print("Done.")

if __name__ == "__main__":
    main()
