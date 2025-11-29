import ast

class NaiveBayesGolf:

    def __init__(self):
        self.freq_classe = {}
        self.freq_atributos = {}
        self.classes = ["Sim", "Nao"]
        self.total = 0

        self.atributos = ["Tempo", "Temperatura", "Umidade", "Vento"]

    def carregar_dados(self, arquivo):
        """
        Lê o arquivo nesse formato:
        dados = [
            ['Chuvoso', 'Quente', 'Alta', 'Fraco', 'Nao'],
            ...
        ]
        """
        with open(arquivo, "r", encoding="utf-8") as f:
            conteudo = f.read()

        # Extrai somente a lista de dados usando ast.literal_eval
        inicio = conteudo.find("[")
        fim = conteudo.rfind("]") + 1
        lista = ast.literal_eval(conteudo[inicio:fim])

        return lista

    def treinar(self, arquivo):
        dados = self.carregar_dados(arquivo)

        # inicializar contagens
        for c in self.classes:
            self.freq_classe[c] = 0
            self.freq_atributos[c] = {att: {} for att in self.atributos}

        # contar frequências
        for registro in dados:
            tempo, temp, umid, vento, jogar = registro

            self.freq_classe[jogar] += 1

            valores = {
                "Tempo": tempo,
                "Temperatura": temp,
                "Umidade": umid,
                "Vento": vento
            }

            for att, valor in valores.items():
                if valor not in self.freq_atributos[jogar][att]:
                    self.freq_atributos[jogar][att][valor] = 0
                self.freq_atributos[jogar][att][valor] += 1

            self.total += 1

    def prob_cond(self, classe, atributo, valor):
        """
        Probabilidade condicional com Laplace smoothing.
        """
        freq_valores = self.freq_atributos[classe][atributo]
        freq = freq_valores.get(valor, 0)
        total_classe = self.freq_classe[classe]

        # número de categorias distintas do atributo
        k = len(freq_valores)

        return (freq + 1) / (total_classe + k)

    def prever(self, exemplo):
        """
        exemplo = {
            "Tempo": "Ensolarado",
            "Temperatura": "Quente",
            "Umidade": "Alta",
            "Vento": "Fraco"
        }
        """

        probs = {}

        for c in self.classes:
            # prior P(C)
            p = self.freq_classe[c] / self.total

            # conditional probabilities
            for att, val in exemplo.items():
                p *= self.prob_cond(c, att, val)

            probs[c] = p

        # normaliza
        s = sum(probs.values())
        for c in probs:
            probs[c] /= s

        classe_final = max(probs, key=probs.get)
        return classe_final, probs


# ==================================
# EXEMPLO DE USO
# ==================================

modelo = NaiveBayesGolf()
modelo.treinar("Base_dados_golfe.txt")

entrada = {
    "Tempo": "Ensolarado",
    "Temperatura": "Ameno",
    "Umidade": "Alta",
    "Vento": "Fraco"
}

classe, probs = modelo.prever(entrada)

print("Entrada:", entrada)
print("Classe prevista:", classe)
print("Probabilidades:")
for c, p in probs.items():
    print(f"P({c}) = {p:.4f}")