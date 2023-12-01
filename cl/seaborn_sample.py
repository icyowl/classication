import matplotlib.pyplot as plt
import seaborn as sns


df = sns.load_dataset('iris')
# print(df.columns)  # ['sepal_length', 'sepal_width', 'petal_length', 'petal_width','species']
# print(set(df['species'])) # {'versicolor', 'setosa', 'virginica'}
sns.set()
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=df)

plt.title("Fisher's Iris")
plt.legend()
plt.show()
