# Passenger Pickup
## Description
This project is a project I did for the course 67842 INTRODUCTION TO ARTIFICIAL INTELLIGENCE in Hebrew University.  
I did this project with Nerya Granot, Nadav Kaduri and Shlomi Adleman.   
The idea originated from the following scenario: Shared Taxi. The differences between a taxi and a bus is that the bus drives from a bus station to another bus station. So usually the passanger will have to walk before and after taking the bus. A taxi can take a passanger from his currrent location to the location he wants to go, but it's more expansive than a bus. So a sweet trade off is when several passangers share seats in a taxi, and each pay their share of the fee. Then they will potensially pay less than a normal taxi, but still enjoy the convenience of a taxi. This idea existed in mainland China before 2018 (DiDiDaChe), and is proven to be a successful business model.  

There are several factors that makes modeling this problem extremely difficult.   
1. Conflict of Interests: If there is only one taxi, and the taxi are going to pick up 4 passangers: each passanger would prefer to get picked up first, and sent to his destination first (minimizing his "waiting time" and "trip time"). Whereas the taxi driver would want to complete the trip in least amount of time, so that he can save gas. (We assume the speed is constant, so the time is equal to the notion of distance.)
2. Economic Consideration: Sometimes the amount a person is willing to wait for a taxi is not constant, it relates to the price of the trip ---- if the price is cheaper, everybody is willing to wait a bit longer. But when the time might also be way too long so it doesn't worth to take taxi anymore. Also the taxi would try to maximize the amount of money he earns with regards to the amount of time he spends.      
3. Multiple Taxis: If there is multiple taxis, how to evenly spread the passangers among each taxi, so that each taxi can take max amount of passangers, and travel least amount of time delivering each passanger. Let's not even mention if each taxi have different amount of seats.
4. Real time planning: sometimes there are new passangers during running time, adding new passangers will change the planning: do we prioritize the new but closer passanger, or do we also care about the passanger that lives far away, but have been waiting for a long time?

Given the scale of the project, we cannot model this complex situations, so we eventually decided to solve a subset of the problem:
So here is the problem settings:  
Let there be n passangers: for each passanger, he/she have a staring point, and a distination point.  
There is one taxi, with k seats.   
We want to plan a route for the taxi, so that the taxi can take these n passangers from their starting point to their destinations.  
The objective of the algorithm is to minimize the length of the entire trip ---- minimizing the drivers driving time, disregard each person's waiting time and trip time, and wthout considering anything about price.    
When there are more passangers than the seat of a taxi, then the taxi will have to take several trips. We will also consider the distance the taxi have to go from trip to trip.  
To further simplify the question: we assume that each taxi have either 1 seat (private taxi) or infinite seats. When the taxi have one seat, this problem is basically planning a route for a private taxi to minimize his time. When a taxi have infinite seats, the problem is to plan a route that sends every passanger to his destination in shortest amount of total time.  
## Solutions
Nerya realized that this problem is very similar to the famous Travelling Salesman Problem (TSP), and the problem is NP complete. So we will need to approach the probelm with hueristics.    
We implemented several solutions to the problem:    
Brute Force  
Greedy Algorithm
Genetic Algorithm  
Local Beam Search  
Hill Climbing  
Simulated Annealing  
I am in charge of the implementation of Genetic Algorithm. The output of our algorithm is the route for the taxi to pick up all the passangers and sent them to their destination.  

## Notes
By the time this readme is written, all of our algorithms, including my Genetic Algorithm is already out of date, I put this idea on github because I think this is an interesting idea, and although this idea have been proven successful in mainlan China, there are still many regions in the world that haven't implemented the shared taxi idea. So if we extend this idea, it might be a potential opprotunity for a start up. 


## How To Run the Project
USER GUIDE:  
option_1:  
run passengers_pickup.py  

option_2:
run passengers_pickup.py from cmd, with the following parameters:  
<capacity> <num_of_passengers> <mode: random or file_path> <algorithm> <method or params>  

capacity = ["1", "inf"]  
num_of_passengers = positive integer  
mode = "-r" for random problem, file_path for import problem from file.  
algorithm = {"G": GENETIC, "HC": HILL_CLIMBING, "SA": SIMULATED_ANNEALING,
              "FC": FIRST_CHOICE, "RR": RESTART_HILL, "B": BEAM, "SB": STOCHASTIC_BEAM, "GREEDY": GREEDY, "T": TABU,
              "LA": LATE_ACCEPTANCE}  
if algorithm == "GREEDY" -> just run  
if algorithm == "GENETIC" -> params = list of 5 parameters. if not given -> default parameters.  
else -> method = {"SI": SWAP_INDEXES, "SA": SWAP_ADJACENT, "IAS": INSERT_AND_SHIFT}  
    
*****
-file format-:  
text file, with num_of_passengers+1 lines.  
the first line contains 2 numbers:  
the first between LONGITUDE_MIN and LONGITUDE_MAX, and the second between LATITUDE_MIN and LATITUDE_MAX  
the rest of num_of_passengers lines contain 4 numbers, each pair must be in the same range like the first line.  

LONGITUDE_MAX = 35.2487  
LONGITUDE_MIN = 35.1694  
LATITUDE_MAX = 31.8062  
LATITUDE_MIN = 31.7493  
*****  
