import sys
import random

class State:
  def __init__(self, position_X, position_Y, isWall, value, isBend, isGoal, action, is_finish, Qtable):
    self.position_X = position_X
    self.position_Y = position_Y
    self.isWall = isWall
    self.value = value
    self.isBend = isBend
    self.isGoal = isGoal
    self.action = action
    self.is_finish = is_finish
    self.Qtable = Qtable

class Q_table:
  def __init__(self, north, east, south, west, bend):
    self.north = north
    self.east = east
    self.south = south
    self.west = west
    self.bend = bend   

def findState(table,i,j):

	for s in table:
		if s.position_X == i and s.position_Y == j:
			return s
	return None		


def DoAction(state,action,table,theta,gamma,M,N,reward_regular,reward_wall,reward_pitfall,reward_goal):
	
	i,j = state.position_X,state.position_Y
	if i<0 and j<0:
		i,j = i*-1,j*-1
	if action == 0:
		new_state = findState(table,i,j+1)
		value = new_state.value
		if new_state.isWall == 0:
			reward = reward_regular
		if new_state.isWall == 1:
			reward = reward_wall	

	if action == 1:
		new_state = findState(table,i+1,j)
		value = new_state.value
		if new_state.isWall == 0:
			reward = reward_regular
		if new_state.isWall == 1:
			reward = reward_wall

	if action == 2:
		new_state = findState(table,i,j-1)
		value = new_state.value
		if new_state.isWall == 0:
			reward = reward_regular
		if new_state.isWall == 1:
			reward = reward_wall

	if action == 3:
		new_state = findState(table,i-1,j)
		value = new_state.value
		if new_state.isWall == 0:
			reward = reward_regular
		if new_state.isWall == 1:
			reward = reward_wall								

	if action == 4:
		new_state = findState(table,-i,-j)
		value = new_state.value
		if new_state.isGoal == 1:
			reward = reward_goal
		if new_state.isGoal == 0:
			reward = reward_pitfall

	return value,reward 	



def Value_Iteration(table,theta,gamma,M,N,reward_regular,reward_wall,reward_pitfall,reward_goal):
	
	v = 0
	v_ = theta+1
	v_diff = 0	

	while v_>theta:
		v_ = -1000
		#print("num")
		
		for state in table:
			v = state.value
			if state.is_finish == 1 or state.isWall == 1:
				continue
			#state.value = state.reward + gamma*nextBestState(table,state,False)
			north_val, nort_rew = DoAction(state,0,table,theta,gamma,M,N,reward_regular,reward_wall,reward_pitfall,reward_goal) 
			east_val, east_rew = DoAction(state,1,table,theta,gamma,M,N,reward_regular,reward_wall,reward_pitfall,reward_goal) 
			south_val, south_rew = DoAction(state,2,table,theta,gamma,M,N,reward_regular,reward_wall,reward_pitfall,reward_goal) 
			west_val, west_rew = DoAction(state,3,table,theta,gamma,M,N,reward_regular,reward_wall,reward_pitfall,reward_goal) 

			act1 = nort_rew + gamma*north_val
			act2 = east_rew + gamma*east_val
			act3 = south_rew + gamma*south_val
			act4 = west_rew + gamma*west_val
			max_value = act1
			state.action = 0
			if act2 > max_value:
				max_value = act2
				state.action = 1
			if act3 > max_value:
				max_value = act3
				state.action = 2
			if act4 > max_value:
				max_value = act4
				state.action = 3	

			if state.isBend == 1:
				bend_val, bend_rew = DoAction(state,4,table,theta,gamma,M,N,reward_regular,reward_wall,reward_pitfall,reward_goal)
				bend_act = bend_rew + gamma*bend_val
				if bend_act > max_value:
					state.action = 4
					max_value = bend_act		

			state.value = max_value		
			if abs(v-max_value) > v_:
				v_ = abs(v-max_value)
	

	return table


def getBestQ(current):

	best = (current.Qtable).north
	action = 0
	if (current.Qtable).east > best:
		best = (current.Qtable).east
		action = 1
	if (current.Qtable).south > best:
		best = (current.Qtable).south
		action = 2
	if (current.Qtable).west > best:
		best = (current.Qtable).west
		action = 3
	if current.isBend == 1 and (current.Qtable).bend > best:
		best = (current.Qtable).bend
		action = 4			
	return action,best

def GoState(action,table,current):
	
	i,j = current.position_X,current.position_Y

	if i<0 and j<0:
		i,j = i*-1,j*-1

	if action == 0:
		new_state = findState(table,i,j+1)
		if new_state.isWall == 1:
			new_state = current	

	elif action == 1:
		new_state = findState(table,i+1,j)
		if new_state.isWall == 1:
			new_state = current

	elif action == 2:
		new_state = findState(table,i,j-1)
		if new_state.isWall == 1:
			new_state = current

	if action == 3:
		new_state = findState(table,i-1,j)
		if new_state.isWall == 1:
			new_state = current								

	if action == 4:
		new_state = findState(table,-i,-j)
		
	return new_state	



def Q_learning(start_table,table,epoc,alpha,gamma,epsilon,M,N,reward_regular,reward_wall,reward_pitfall,reward_goal):
	
	

	for i in range(0,epoc):
		current = start_table[random.randint(0,len(start_table)-1)]
		exp_rate = epsilon 
		#print(i)

		while current.is_finish != 1:
			
			decision = random.choices(population=["explore","policy"],weights=[exp_rate,1-exp_rate])
			#print(decision)
			if decision == ["explore"]:
				#print(current.position_X,current.position_Y)	
				if current.isBend == 1:
					action = random.randint(0,4)
				else:
					action = random.randint(0,3)

				new_state = GoState(action,table,current)
				_,reward = DoAction(current,action,table,0,gamma,M,N,reward_regular,reward_wall,reward_pitfall,reward_goal)	
				
				#print(action,"-->",new_state.position_X,new_state.position_Y)
				_,Q_max = getBestQ(new_state)

				if action == 0:
					(current.Qtable).north = (1-alpha)*(current.Qtable).north + alpha*(reward + gamma*Q_max) 
				if action == 1:
					(current.Qtable).east = (1-alpha)*(current.Qtable).east + alpha*(reward + gamma*Q_max)
				if action == 2:
					(current.Qtable).south = (1-alpha)*(current.Qtable).south + alpha*(reward + gamma*Q_max)
				if action == 3:
					(current.Qtable).west = (1-alpha)*(current.Qtable).west + alpha*(reward + gamma*Q_max)
				if action == 4:
					(current.Qtable).bend = (1-alpha)*(current.Qtable).bend + alpha*(reward + gamma*Q_max)
					
				current = new_state	


			elif decision == ["policy"]:

				if current.isBend == 1:
					action = random.randint(0,4)
				else:
					action = random.randint(0,3)
				#print(current.position_X,current.position_Y)		
				new_state = GoState(action,table,current)
				_,reward = DoAction(current,action,table,0,gamma,M,N,reward_regular,reward_wall,reward_pitfall,reward_goal)	
				
				#print(action,"-->",new_state.position_X,new_state.position_Y, reward)
				_,Q_max = getBestQ(new_state)
				
				if action == 0:
					(current.Qtable).north = (1-alpha)*(current.Qtable).north + alpha*(reward + gamma*Q_max) 
				if action == 1:
					(current.Qtable).east = (1-alpha)*(current.Qtable).east + alpha*(reward + gamma*Q_max)
				if action == 2:
					(current.Qtable).south = (1-alpha)*(current.Qtable).south + alpha*(reward + gamma*Q_max)
				if action == 3:
					(current.Qtable).west = (1-alpha)*(current.Qtable).west + alpha*(reward + gamma*Q_max)
				if action == 4:
					(current.Qtable).bend = (1-alpha)*(current.Qtable).bend + alpha*(reward + gamma*Q_max)
					
					
				#print(current.position_X,current.position_Y, current.Qtable.north, current.Qtable.east, current.Qtable.south, current.Qtable.west, current.Qtable.bend)
				current = new_state	



	return table
	


if len(sys.argv) != 3:
	sys.exit( "usage: python command_line_argument.py <name>")

name=sys.argv[1] 
name2=sys.argv[2] 



with open(name,'r') as file_data:
	lines = file_data.readlines()
	function = lines[0].strip()
	
#####################^V^#########################

	if function == "V":
		flag = True	
		theta = float(lines[1].strip())
		gamma = float(lines[2].strip())

		ln = lines[3].strip()
		for i in range(0,len(ln)):
			if ln[i] == " ":
				indx = i
				break
		
		M,N = int(ln[:i]), int(ln[i+1:])
		
	#################### OBJECT TAKE ###########################
				
		obs_number = int(lines[4].strip())
		obs_place = []

		for i in range(0,obs_number):
			ln = lines[5+i].strip()
			
			for j in range(0,len(ln)):
				if ln[j] == " ":
					indx = j
					break

			first = ln[:indx]
			second = ln[indx+1:]	
				
			if first[0] == "-":
				num1 = int(first[1:]) *-1
			else:	
				num1 = int(first[0:])
				
			if second[0] == "-":
				num2 = int(second[1:]) *-1
			else:	
				num2 = int(second[0:])

			obs_place.append((num1, num2))

	##################### PIT TAKE ##########################
    	
		pit_number = int(lines[5+obs_number].strip())
		pit_place = []

		for i in range(0,pit_number):

			ln = lines[6+obs_number+i].strip()
			
			for j in range(0,len(ln)):
				if ln[j] == " ":
					indx = j
					break

			first = ln[:indx]
			second = ln[indx+1:]	

			if first[0] == "-":
				num1 = int(first[1:]) 
			else:	
				num1 = int(first[0:]) *-1
				
			if second[0] == "-":
				num2 = int(second[1:]) 
			else:	
				num2 = int(second[0:]) *-1

			pit_place.append((num1, num2))


	##################### GOAL STATE TAKE ###################################
			
		ln = lines[6+obs_number+pit_number].strip()	
			
		for j in range(0,len(ln)):
			if ln[j] == " ":
				indx = j
				break

		first = ln[:indx]
		second = ln[indx+1:]	
			
		if first[0] == "-":
			num1 = int(first[1:]) 
		else:	
			num1 = int(first[0:]) *-1
				
		if second[0] == "-":
			num2 = int(second[1:]) 
		else:	
			num2 = int(second[0:]) *-1

		goal = (num1, num2) 	

	##################### REWARD TAKE ###################################

		ln = lines[6+obs_number+pit_number+1].strip()
		for j in range(0,len(ln)):
			if ln[j] == " ":
				indx = j
				break	

		first = ln[:indx]
		ln = ln[indx+1:]	
				
		if first[0] == "-":
			num1 = float(first[1:]) *-1
		else:	
			num1 = float(first[0:]) 		
		reward_regular = num1

		for j in range(0,len(ln)):
				if ln[j] == " ":
					indx = j
					break	

		first = ln[:indx]
		ln = ln[indx+1:]	
				
		if first[0] == "-":
			num1 = float(first[1:]) *-1
		else:	
			num1 = float(first[0:]) 	
		reward_obj = num1

		for j in range(0,len(ln)):
				if ln[j] == " ":
					indx = j
					break	

		first = ln[:indx]
		ln = ln[indx+1:]	
				
		if first[0] == "-":
			num1 = float(first[1:]) *-1
		else:	
			num1 = float(first[0:]) 	
		reward_pitfall = num1		

		for j in range(0,len(ln)):
				if ln[j] == " ":
					indx = j
					break

		first = ln[:indx]
		ln = ln[indx+1:]	
				
		if first[0] == "-":
			num1 = float(first[1:]) *-1
		else:	
			num1 = float(first[0:]) 	
		reward_goal = num1	



######################^Q^########################

	if function == "Q":
		flag = False
		epoc = int(lines[1].strip())	
		alpha = float(lines[2].strip())
		gamma = float(lines[3].strip())
		epsilon = float(lines[4].strip())
		
		ln = lines[5].strip()
		for i in range(0,len(ln)):
			if ln[i] == " ":
				indx = i
				break
		
		M,N = int(ln[:i]), int(ln[i+1:])

	####################OBJE TAKE########################3
		obs_number = int(lines[6].strip())
		obs_place = []

		for i in range(0,obs_number):
			ln = lines[7+i].strip()
			
			for j in range(0,len(ln)):
				if ln[j] == " ":
					indx = j
					break

			first = ln[:indx]
			second = ln[indx+1:]	
				
			if first[0] == "-":
				num1 = int(first[1:]) *-1
			else:	
				num1 = int(first[0:])
				
			if second[0] == "-":
				num2 = int(second[1:]) *-1
			else:	
				num2 = int(second[0:])

			obs_place.append((num1, num2))

	##################### PIT TAKE ##########################
    	
		pit_number = int(lines[7+obs_number].strip())
		pit_place = []

		for i in range(0,pit_number):

			ln = lines[8+obs_number+i].strip()
			
			for j in range(0,len(ln)):
				if ln[j] == " ":
					indx = j
					break

			first = ln[:indx]
			second = ln[indx+1:]	

			if first[0] == "-":
				num1 = int(first[1:]) 
			else:	
				num1 = int(first[0:]) *-1
				
			if second[0] == "-":
				num2 = int(second[1:]) 
			else:	
				num2 = int(second[0:]) *-1

			pit_place.append((num1, num2))

	##################### GOAL STATE TAKE ###################################
			
		ln = lines[8+obs_number+pit_number].strip()	
			
		for j in range(0,len(ln)):
			if ln[j] == " ":
				indx = j
				break

		first = ln[:indx]
		second = ln[indx+1:]	
			
		if first[0] == "-":
			num1 = int(first[1:]) 
		else:	
			num1 = int(first[0:]) *-1
				
		if second[0] == "-":
			num2 = int(second[1:]) 
		else:	
			num2 = int(second[0:]) *-1

		goal = (num1, num2) 				


	##################### REWARD TAKE ###################################

		ln = lines[8+obs_number+pit_number+1].strip()
		for j in range(0,len(ln)):
			if ln[j] == " ":
				indx = j
				break	

		first = ln[:indx]
		ln = ln[indx+1:]	
				
		if first[0] == "-":
			num1 = float(first[1:]) *-1
		else:	
			num1 = float(first[0:]) 		
		reward_regular = num1

		for j in range(0,len(ln)):
				if ln[j] == " ":
					indx = j
					break	

		first = ln[:indx]
		ln = ln[indx+1:]	
				
		if first[0] == "-":
			num1 = float(first[1:]) *-1
		else:	
			num1 = float(first[0:]) 	
		reward_obj = num1

		for j in range(0,len(ln)):
				if ln[j] == " ":
					indx = j
					break	

		first = ln[:indx]
		ln = ln[indx+1:]	
				
		if first[0] == "-":
			num1 = float(first[1:]) *-1
		else:	
			num1 = float(first[0:]) 	
		reward_pitfall = num1		

		for j in range(0,len(ln)):
				if ln[j] == " ":
					indx = j
					break

		first = ln[:indx]
		ln = ln[indx+1:]	
				
		if first[0] == "-":
			num1 = float(first[1:]) *-1
		else:	
			num1 = float(first[0:]) 	
		reward_goal = num1	

start_table = []
table = []
for j in range(0,N+2):	
	for i in range(0,M+2):
	
		if (j,i) == goal:
			table.append(State(j,i,0,0,1,0,None,0,Q_table(0,0,0,0,0)))
			table.append(State(-j,-i,reward_goal,0,0,1,None,1,Q_table(0,0,0,0,0)))
		elif (j,i) in obs_place:
			table.append(State(j,i,1,0,0,0,None,0,Q_table(0,0,0,0,0)))
		elif (j,i) in pit_place:
			table.append(State(-j,-i,0,0,0,0,None,0,Q_table(0,0,0,0,0)))
			table.append(State(j,i,0,0,1,0,None,0,Q_table(0,0,0,0,0)))
		elif j == 0 or i == 0:
			table.append(State(j,i,1,0,0,0,None,0,Q_table(0,0,0,0,0)))
		elif j == N+1 or i == M+1:
			table.append(State(j,i,1,0,0,0,None,0,Q_table(0,0,0,0,0)))
		else:
			tmp = State(j,i,0,0,0,0,None,0,Q_table(0,0,0,0,0))
			table.append(tmp)
			start_table.append(tmp)
			
			

if flag:
	sol_table = Value_Iteration(table,theta,gamma,M,N,reward_regular,reward_obj,reward_pitfall,reward_goal)
	#Policy_Writing(table,M,N)	
	with open(name2,'w') as outfile:
		for state in sol_table:
			if state.isWall == 0:
				outfile.write(str(state.position_X) +" "+ str(state.position_Y) +" "+ str(state.action) + "\n" )


else:
	sol_table = Q_learning(start_table,table,epoc,alpha,gamma,epsilon,M,N,reward_regular,reward_obj,reward_pitfall,reward_goal)	
	with open(name2,'w') as outfile:
		for state in sol_table:
			if state.isWall == 0:
				action,_ = getBestQ(state)
				outfile.write(str(state.position_X) +" "+ str(state.position_Y) +" "+ str(action) + "\n" )


