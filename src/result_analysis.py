# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score

# %%
df_orig = pd.read_csv("../result/final_result.csv")

# %%
print("Regular Expression")
print(accuracy_score(df_orig['label'], df_orig['pred_re']))
print(balanced_accuracy_score(df_orig['label'], df_orig['pred_re']))
print(cohen_kappa_score(df_orig['label'], df_orig['pred_re'], weights='quadratic'))
# %%
print("\nBERT")
print(accuracy_score(df_orig['label'], df_orig['pred_class']))
print(balanced_accuracy_score(df_orig['label'], df_orig['pred_class']))
print(cohen_kappa_score(df_orig['label'], df_orig['pred_class'], weights='quadratic'))

# %%

result = {
    'model':["BERT", "BERT", "BERT", "RE", "RE", "RE"],
    'metric':["Accuracy", "Balanced Accuracy", 'Quadratic Kappa', "Accuracy", "Balanced Accuracy", 'Quadratic Kappa'],
    'value':[
        accuracy_score(df_orig['label'], df_orig['pred_class']),
        balanced_accuracy_score(df_orig['label'], df_orig['pred_class']),
        cohen_kappa_score(df_orig['label'], df_orig['pred_class'], weights='quadratic'),
        
        accuracy_score(df_orig['label'], df_orig['pred_re']), 
        balanced_accuracy_score(df_orig['label'], df_orig['pred_re']),
        cohen_kappa_score(df_orig['label'], df_orig['pred_re'], weights='quadratic'),
    ],
}

df_result = pd.DataFrame(result)
df_result = df_result.assign(
    value = lambda x: x['value']*100
)

# %%

def plot_compusion_matrix(ax, data, x, y, title):
    
    absolute_values = pd.crosstab(data[x], data[y]).values
    percentage_values = pd.crosstab(data[x], data[y], normalize='columns').values * 100
    
    sns.heatmap(
        percentage_values, 
        annot=True, 
        ax=ax, 
        cbar=True,
        vmin=0,
        vmax=100,
        annot_kws={'fontsize':20}, 
        cmap="Blues", 
        xticklabels=["Absent", "Mild", "Moderate", "Severe"],
        yticklabels=["Absent", "Mild", "Moderate", "Severe"],
    )
    
    percentage_values = percentage_values.reshape(-1)
    absolute_values = absolute_values.reshape(-1)
    
    for idx, t in enumerate(ax.texts): 
        t.set_text(f"{absolute_values[idx]:3d}\n({percentage_values[idx]:2.2f}%)")
        
    ax.set_xlabel("Ground Truth", fontsize=25)
    ax.set_ylabel("Prediction", fontsize=25)
    ax.tick_params(labelsize=25)
    ax.set_title(title, fontsize=30)
    ax.collections[0].colorbar.ax.tick_params(labelsize=25)
    
    return ax

# %%
fig, ax = plt.subplots(1, 1, figsize=(11, 10), facecolor='w')
ax = plot_compusion_matrix(ax=ax, data=df_orig, x='label', y='pred_class', title='BERT')
plt.tight_layout()
plt.savefig("../result/fig/bert_confusion_matrix.png", dpi=300)
plt.show()

# %%
fig, ax = plt.subplots(1, 1, figsize=(11, 10), facecolor='w')
ax = plot_compusion_matrix(ax=ax, data=df_orig, x='label', y='pred_re', title='Regular Expression')
plt.tight_layout()
plt.savefig("../result/fig/re_confusion_matrix.png", dpi=300)
plt.show()
# %%

def plot_bar_result(dataframe, x, hue, file_nm, dpi=500):
    
    x_order = ["Accuracy", "Balanced Accuracy", "Quadratic Kappa"]
    hue_order = ["RE", "BERT"]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='w')
    
    fig_a = sns.barplot(
        x=x, 
        y="value", 
        ax=ax, 
        hue=hue, 
        data=dataframe, 
        order=x_order,
        hue_order=hue_order, 
        linewidth=5,  
        edgecolor=".2", 
        palette=["#EEC374", "#E63E62"]
    )
    
    fig_a.legend(fontsize=16)
    
    ax.set_ylabel("Score", fontsize=30)
    ax.set_xlabel("", fontsize=30)
    ax.tick_params(labelsize=20)
    
    plt.ylim(75, 105)
    plt.tight_layout()
    plt.savefig(file_nm, dpi=dpi)
    plt.show()
    
    return None
# %%
plot_bar_result(
    dataframe=df_result,
    x='metric',
    hue='model',
    file_nm="../result/fig/bar_plot.png"
)
# %%
