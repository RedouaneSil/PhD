#FBSDE solver

##How to run the code ?
Open the main file.
Every parameter can be change at the beginning of the main file

###Stochastic simulations
You can modify the Lambda parameter to set the number of Brownian motions for the least square problem.
####Simulation with one set of parameter : 
either you modify alpha, beta and gamma at the beginning of the code, compile the code and type simulation()
or you directly type simulation(alpha,beta,gamma)

####Simulation with fixed beta/gamma and alpha changing : 
either you modify vector_alpha which is the vector of alpha values that program will be simulating, beta and gamma at the beginning of the code, compile the code and type simulation_alpha()
or you directly type simulation_alpha(vector_alpha,beta,gamma)

####Simulation with fixed alpha/gamma and beta changing : 
either you modify vector_beta which is the vector of beta values that the program will be simulating with, alpha and gamma at the beginning of the code, compile the code and type simulation_alpha()
or you directly type simulation_beta(alpha,vector_beta,gamma)

####Simulation with fixed alpha/gamma and gamma changing : 
either you modify vector_gamma which is the vector of gamma values that the program will be simulating with, alpha and beta at the beginning of the code, compile the code and type simulation_alpha()
or you directly type simulation_beta(alpha,beta,vector_gamma)

###Deterministic simulation

####Simulation with one set of parameter : 
either you modify alpha, beta and gamma at the beginning of the code, compile the code and type deterministic_sim()
or you directly type deterministic_sim(alpha,beta,gamma)

####Simulation with fixed beta/gamma and alpha changing : 
either you modify vector_alpha which is the vector of alpha values that program will be simulating, beta and gamma at the beginning of the code, compile the code and type deterministic_sim_alpha()
or you directly type deterministic_sim_alpha(vector_alpha,beta,gamma)

####Simulation with fixed alpha/gamma and beta changing : 
either you modify vector_beta which is the vector of beta values that the program will be simulating with, alpha and gamma at the beginning of the code, compile the code and type deterministic_sim_beta()
or you directly type deterministic_sim_beta(alpha,vector_beta,gamma)
