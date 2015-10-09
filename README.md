#GENETIC PROGRAMMING FOR SYMBOLIC REGRESSION
Author: Pedro Ramos
email: phramoscosta@gmail.com

-> This program was developed as part of a coursework for the module Natural Computing (Computacao Natural) at the Federal University of Minas Gerais (UFMG). If you need more details of it just email me. It is very incomplete due to the time I had to do this task (about 10 days) so I couldnt make a better code. Since I had tons of other courseworks to do by then, I just let this one go. So feel free to take a look and change it as you please.

HOW IT WORKS:
-> You get a dataset, a txt file with a set of coordinates. The program uses geneti programming to find the function that best suits that set of coordinates.

TO RUN:
-> It requires python 2.7 or higher.

-> The main module is gp_main.py. To run it just 

$ python gp_main.py

CHANGING THE INPUT DATASET AND OTHER PARAMETERS:
-> To change the input file, just open gp_main.py and change the global variable FNAME. It is recommended that all input datasets be stored in the /datasets folder. It is set to a default one.

-> To change any parameters of the genetic program, just take a look at gp_main.py . The parameters are globals. The implementation is well commented and explained there.

DETAILS OF IMPLEMENTATION:
-> Details of implementation can be found in portuguese in the PDF document tp1_doc_aluno_2011049436.pdf 

-> Hard coding details of each module can be found inside them. There are three modules:

-> -> expression_tree.py: it describes the structure used for expression trees, that represent the individuals

-> -> individual.py: it is merely a top abstraction layer, that implements the operands (mutation and crossover) as well as the fitness function parameter mechanism in case there is any need of changing the fitness function.

-> -> gp_main.py: the only module that you need to look at if you want to change anything (fitness function, genetic program parameters and etc). 

TAKE A LOOK AT THEM, they are well commented.
