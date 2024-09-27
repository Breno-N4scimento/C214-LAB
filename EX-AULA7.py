import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class ClassificadorKNN:
    def __init__(self, modelo, encoder, test_size=0.2, random_state=42):
        self.modelo = modelo
        self.encoder = encoder
        self.test_size = test_size
        self.random_state = random_state

    def carregar_dados(self, caminho_csv):
        # Carrega os dados do CSV
        data = pd.read_csv(caminho_csv)
        return data

    def preparar_dados(self, data):
        # Aplica o LabelEncoder em todas as colunas categóricas, exceto na coluna target 'Disorder'
        for column in data.columns[:-1]:
            data[column] = self.encoder.fit_transform(data[column])

        # Codifica a variável target (Disorder)
        data['Disorder'] = self.encoder.fit_transform(data['Disorder'])

        # Divide os dados em features (X) e rótulos (y)
        X = data.drop(columns=['Disorder'])  # Todas as colunas, exceto a última
        Y = data['Disorder']  # A última coluna (saída)

        # Divide os dados em conjunto de treino e teste
        xTreino, xTeste, yTreino, yTeste = train_test_split(X, Y, test_size=self.test_size, random_state=self.random_state)

        return xTreino, xTeste, yTreino, yTeste

    def treinar_modelo(self, xTreino, yTreino):
        # Treina o modelo
        self.modelo.fit(xTreino, yTreino)

    def testar_modelo(self, xTeste, yTeste):
        return self.modelo.score(xTeste, yTeste)

modeloKNN = KNeighborsClassifier()
convertVar = LabelEncoder()

classificador = ClassificadorKNN(modeloKNN, convertVar)

data = classificador.carregar_dados(r"C:\Users\elbri\OneDrive\Área de Trabalho\psicoFETIN\dataset.csv")

xTreino, xTeste, yTreino, yTeste = classificador.preparar_dados(data)

classificador.treinar_modelo(xTreino, yTreino)

precisao = classificador.testar_modelo(xTeste, yTeste)
print(f"Acurácia do modelo: {precisao}")
