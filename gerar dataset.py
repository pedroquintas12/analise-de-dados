import pandas as pd
import numpy as np
import random
from faker import Faker

# Informações base
series_turmas = {
    "1º ano": {"A": 30, "B": 34, "C": 28, "D": 32, "E": 25},
    "2º ano": {"A": 29, "B": 30, "C": 32, "D": 27},
    "3º ano": {"A": 32, "B": 30, "C": 31}
}

disciplinas = [
    "Matemática", "Português", "História", "Geografia", "Biologia",
    "Química", "Física", "Educação Física", "Artes", "Filosofia",
    "Sociologia", "Inglês"
]

generos = ["Masculino", "Feminino", "Outro"]
etnias = ["Branco", "Negro", "Pardo", "Indígena", "Amarelo"]
orientacoes = ["Heterossexual", "Homossexual", "Bissexual", "Assexual", "Outro"]
faixa_renda = ["Até 1 SM", "1 a 2 SM", "2 a 5 SM", "5 a 10 SM", "Mais de 10 SM"]

# Geração do dataset
alunos = []
id_counter = 1

# Inicializa o gerador de nomes no idioma português do Brasil
fake = Faker('pt_BR')

# Gera uma lista com 360 nomes
nomes = [{"Nome": f"{fake.first_name()} {fake.last_name()}"} for _ in range(360)]

# Cria o DataFrame
df_nomes = pd.DataFrame(nomes)

for serie, turmas in series_turmas.items():
    for turma, qtd in turmas.items():
        for i in range(qtd):
            notas = {f"Nota {disc}": round(np.random.uniform(0, 10), 2) for disc in disciplinas}
            media_geral = round(np.mean(list(notas.values())), 2)

            # Cálculo da média das disciplinas de exatas
            notas_exatas = [notas[f"Nota {disc}"] for disc in ["Matemática", "Física", "Química"]]
            media_exatas = np.mean(notas_exatas)

            # Definir frequência com base na média das disciplinas de exatas
            if media_exatas < 3:
                frequencia = round(np.random.uniform(20, 50), 2)
            elif media_exatas < 6:
                frequencia = round(np.random.uniform(40, 80), 2)
            else:
                frequencia = round(np.random.uniform(70, 100), 2)

            aluno = {
                "ID": id_counter,
                "Nome": nomes[i]["Nome"],
                "Ano Letivo": 2025,
                "Série": serie,
                "Turma": turma,
                "Frequência (%)": frequencia,
                **notas,
                "Média Geral no ano": media_geral,
                "Gênero": random.choice(generos),
                "Etnia": random.choice(etnias),
                "Idade": random.randint(15, 18),
                "Orientação Sexual": random.choice(orientacoes),
                "Faixa de Renda": random.choice(faixa_renda)
            }
            alunos.append(aluno)
            id_counter += 1

# Criar DataFrame e salvar em Excel
df = pd.DataFrame(alunos)
df.to_excel("dataset_alunos.xlsx", index=False)
print("Arquivo 'dataset_alunos.xlsx' criado com sucesso.")
