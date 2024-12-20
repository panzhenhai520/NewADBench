import matplotlib.pyplot as plt
models = ['LR', 'NB', 'SVM', 'MLP', 'RF', 'LGB', 'XGB', 'CatB', 'Pytorch_LSTM', 'Pytorch_GRU', 'keras_lstm_model', 'SVM_model', 'RandomForest_model', 'ResNet', 'FTTransformer']
datasets = ['25_musk', '3_backdoor', '9_census']


fit_time_data = [
    [0.020915508270263672, 0.0060002803802490234, 0.25148487091064453, 0.4099245071411133, 0.3011515140533447, 0.31530141830444336, 0.23125338554382324, 20.343641996383667, 36.05746841430664, 23.608179807662964, 682.669370174408, 0.0440676212310791, 0.32607030868530273, 27.80200982093811, 284.9662232398987],
    [0.13714265823364258, 0.02139568328857422, 4.174344301223755, 7.448667764663696, 1.4946720600128174, 0.2867705821990967, 1.5475268363952637, 10.442196130752563, 149.22236514091492, 75.02923917770386, 2954.6051557064056, 1.4934208393096924, 0.7903933525085449, 104.22162008285522, 2805.2853972911835],
    [0.3197035789489746, 0.048601388931274414, 21.311525583267212, 13.59339714050293, 1.802400827407837, 0.35819149017333984, 5.103503704071045, 10.558105707168579, 112.28148627281189, 70.44844150543213, 9256.53019952774, 17.342479705810547, 1.1768126487731934, 80.89642882347107, 13494.06303858757]
]

plt.figure(figsize=(12, 6))
for i in range(len(datasets)):
    plt.plot(models, fit_time_data[i], marker='o', label=datasets[i])
highlighted_models = ['Pytorch_LSTM', 'Pytorch_GRU', 'keras_lstm_model', 'SVM_model', 'RandomForest_model']
legend_labels = [f'\\textcolor{{red}}{{{model}}}' if model in highlighted_models else model for model in datasets]

plt.title('Fit Time Comparison')
plt.xlabel('Models')
plt.ylabel('Fit Time (seconds)')
plt.xticks(rotation=45)
plt.legend(labels=legend_labels)
plt.tight_layout()
plt.show()