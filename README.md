https://github.com/MehdiShahbazi/Webots-reinforcement-navigation/blob/master/controllers/REINFORCE/REINFORCE.py


Para converter um espaço de estado contínuo em um espaço de estado discreto, que é uma prática comum em algoritmos de aprendizado por reforço como o Q-Learning, utilizamos um processo chamado discretização. Isso é especialmente útil quando lidamos com variáveis ambientais que podem assumir uma gama infinita ou muito ampla de valores, tornando-se impraticável manter uma tabela Q ou função valor para cada possível estado.

Passos para a Discretização
Definição de Bins:

Primeiramente, define-se uma série de "bins" ou intervalos para cada dimensão do espaço de estado. Por exemplo, se temos uma variável de estado que mede a temperatura, que pode variar de 0 a 100 graus, podemos dividir essa faixa em intervalos de 10 graus cada (0-10, 11-20, ..., 91-100).
Cada bin representa uma categoria ou um valor discreto dentro do qual todos os valores contínuos reais serão agrupados.
Mapeamento dos Valores Contínuos para os Bins:

Quando um valor contínuo precisa ser avaliado, ele é mapeado ao seu bin correspondente. Isso é frequentemente feito usando funções como np.digitize() em Python, que determina o índice do bin ao qual o valor pertence.
Por exemplo, se a temperatura for 37 graus, ela será mapeada para o bin correspondente ao intervalo 31-40.
Uso dos Índices dos Bins:

Em vez de usar o valor contínuo real, usamos o índice do bin (ou o valor representativo do bin, como o ponto médio) para todas as operações subsequentes, incluindo atualizações da tabela Q em algoritmos de Q-Learning.
Isso simplifica significativamente o número de estados possíveis que o algoritmo precisa gerenciar, permitindo o uso de técnicas de aprendizado por reforço que normalmente requerem estados discretos.
Considerações Importantes
Escolha do Número de Bins:

O número de bins é crucial e pode afetar significativamente a performance do algoritmo de aprendizado. Muitos bins podem levar a uma tabela Q muito granular com muitos estados raramente visitados, enquanto poucos bins podem causar uma perda significativa de informação importante sobre o estado.
Adaptação Dinâmica:

Em alguns casos, pode ser útil adaptar dinamicamente o número ou o tamanho dos bins com base na experiência adquirida durante o treinamento, uma técnica conhecida como quantização adaptativa.
Generalização:

A discretização também pode ser combinada com técnicas de generalização, onde estados similares são tratados de maneira semelhante, o que pode ajudar na convergência do aprendizado.
Utilizar a discretização para transformar espaços de estado contínuos em discretos é uma maneira eficaz de aplicar algoritmos de aprendizado por reforço a problemas que, de outra forma, seriam inabordáveis devido à complexidade ou ao tamanho do espaço de estado.





2. Definir state_limits
state_limits define os limites mínimos e máximos para cada dimensão do seu espaço de estado. Por exemplo, se você está trabalhando com um robô que tem posições que variam de 0 a 5 metros em uma dimensão, você definiria os limites dessa dimensão como (0, 5).

Coleta de Dados: Uma maneira de determinar esses limites é observar o comportamento do ambiente (ou simulador) em várias situações ou consultar a documentação técnica para encontrar esses limites.
Exemplo Prático: Se você está medindo a distância até um objetivo e sabe que a distância máxima possível é 5 metros, seus limites podem ser de (0, 5) para essa dimensão.
3. Definir num_bins
num_bins determina quantos intervalos discretos (bins) você deseja para cada dimensão do estado. Isso influencia a granularidade com que você modela o espaço de estado:

Maior Granularidade: Mais bins permitem uma modelagem mais detalhada do ambiente, mas aumentam o tamanho da tabela Q e podem requerer mais tempo e dados para aprender eficazmente.

Menor Granularidade: Menos bins simplificam o modelo e podem acelerar a aprendizagem, mas com o custo de possivelmente perder detalhes importantes sobre o estado.

Escolha Prática: Uma regra geral é começar com um número menor de bins (por exemplo, 10 bins por dimensão) e ajustar com base no desempenho do agente. Se o agente parece não aprender políticas eficazes, pode ser necessário ajustar o número de bins ou os próprios limites.
