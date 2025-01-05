from bertopic import BERTopic

# Exemplo de textos para treino
texts = [
    # Terremotos
    "Tremor de terra foi sentido na região metropolitana de Tóquio, sem relatos de danos maiores.",
    "Sismo de magnitude 7.0 abalou a costa do Chile, deixando várias casas destruídas.",
    
    # Inundações
    "Chuvas intensas causaram inundações em várias partes de Mumbai, paralisando o transporte público.",
    "O rio transbordou em uma pequena cidade da Alemanha, forçando centenas de pessoas a evacuarem suas casas.",
    
    # Incêndios Florestais
    "Incêndio florestal se espalha rapidamente no norte da Califórnia, ameaçando comunidades locais.",
    "Bombeiros lutam para controlar incêndio na Amazônia, que já destruiu milhares de hectares de floresta.",
    
    # Furacões
    "Furacão Katrina devastou Nova Orleans, deixando uma trilha de destruição e desabrigando milhares.",
    "Furacão Maria causa estragos em Porto Rico, interrompendo serviços básicos e causando apagões.",
    
    # Deslizamentos de Terra
    "Deslizamento de terra bloqueia rodovia principal em Nepal, dificultando o acesso das equipes de resgate.",
    "Fortes chuvas resultaram em um grande deslizamento de terra em uma aldeia na Indonésia, soterrando várias casas.",
    
    # Tsunamis
    "Um tsunami gerado por um terremoto no Oceano Índico atingiu a costa da Indonésia, causando destruição em larga escala.",
    "Alerta de tsunami foi emitido após terremoto em Fiji, mas não houve danos significativos relatados.",
    
    # Secas
    "Seca prolongada no Sahel causa crise humanitária, com milhares de pessoas em risco de fome.",
    "Reservatórios estão secando em São Paulo devido à falta de chuvas, levando a racionamento de água."
]


# Treinar o modelo
topic_model = BERTopic()
topic_model.fit(texts)

# Salvar o modelo treinado
topic_model.save("/workspaces/noah-chatbot/bertopic/models/bertopic_model")
