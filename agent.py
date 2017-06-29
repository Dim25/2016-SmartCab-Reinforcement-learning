import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
#
# from awesome_print import ap
import random
import numpy as np
import time
import sys

CSI="\x1B["
reset=CSI+"m"

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Q = {}

        # Discount
        # self.gamma = 0.8
        self.gamma = 0.5
        # self.gamma = 0.1
        # self.gamma = 0.5

        self.epsilon = 0.95 # <===
        # self.epsilon = 0.75

        # self.epsilon = 0.5

        # self.epsilon = 0.5
        # self.epsilon = 0.05
        # self.epsilon = 0.25
        # self.epsilon = 0.99
        # self.epsilon = 0.95

        # Learning Rate
        # self.alpha = 1.
        # self.alpha = 0.8  # <===
        self.alpha = 0.5

        # self.alpha = 0.25
        # self.alpha = 0.5

        self.trips_rewards = {}
        self.trips_timers = {}
        self.trips_steps = {}
        self.trips_finished = {}

        self.trip_reward = 0
        self.total_reward = 0
        self.total_finished = 0
        
        self.current_step = 1
        self.total_steps = 1
        self.trip_number = 0

        # self.max_trips = 20
        self.max_trips = 100

        self.valid_actions = [None, 'forward', 'left', 'right']

        print ''
        print CSI+"7;30;43m" + "Welcome to SmartCab simulation" + CSI + "0m"
        print ''
        print '=================================================='
        print '-------------------- config ----------------------'
        print 'gamma: ' + str(self.gamma)
        print 'epsilon: ' + str(self.epsilon)
        print 'alpha: ' + str(self.alpha)
        print '--------------------------------------------------'
        print ''

        self.timer = time.time()

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required        
        self.trip_number += 1
        self.current_step = 0
        self.trip_reward = 0        
        self.trip_begin_time = time.time()

        if self.trip_number > self.max_trips: # 20       
            # Recording results to log.txt
            print 'self.results_log'
            print self.results_log

            # # Output to the log file
            # f = open('log.txt','ab+')
            # f.write(self.results_log)
            # f.write("\n")
            # f.close()

            # sys.exit(0) # exit if reached the self.max_trips from config

        print '[SmartCab] New trip - awesome!'        

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        state = {}
        state['action'] = self.next_waypoint # desired action
        state['light'] = inputs['light']
        state['oncoming_traffic'] = inputs['oncoming']
        state['traffic_on_the_left'] = inputs['left']
        state = tuple(state.items())        
        self.state = state

        print ''
        print '=================================================='
        print '-------------------- stats -----------------------'
        print 'trip_number: ' + str(self.trip_number)
        print 'current_step: ' + str(self.current_step)
        print 'deadline: ' + str(deadline)
        print 'total_steps: ' + str(self.total_steps)
        print 'trip_reward: ' + str(self.trip_reward)
        print 'total_reward: ' + str(self.total_reward)
        print 'total_reward/total_steps: ' + str(self.total_reward/self.total_steps)
        print 'timer: ' + str(round(time.time() - self.timer,2)) 
        print '-------------------- STATE -----------------------'        
        # ap(state)
        print state
        print '--------------------------------------------------'

        # TODO: Select action according to your policy
        # Select action based on previous outcomes

        # Check if this state occured before        
        print '[SmartCab] Check if this state occured before'
        if state in self.Q:
            print '=> State is found in Q-table'
            print '=> State occured ' + str(self.Q[state]['count']) + ' time(s)'

            print '[SmartCab] Compare the value of pre configurated parameter epsilon with random value: '
            random_value = random.random()
            
            if random_value > self.epsilon:
                print '=> random_value: ' + str(random_value) + ' > epsilon: ' + str(self.epsilon)
                action = random.choice( self.valid_actions )
                print '=> Taking random action: ' + str(action)
            else:
                print '=> random_value: ' + str(random_value) + ' < epsilon: ' + str(self.epsilon)            
                print '[SmartCab] Find available_actions with maximum Q^ score:'
                available_actions = [ k for k,v in self.Q[state]['actions_Q^'].items() if v == max(self.Q[state]['actions_Q^'].values()) ]
                print '=> ' + str(available_actions)
                action = random.choice( available_actions )
                
                print '[SmartCab] Take random action from the list of available_actions with max Q^: ' + str(action)

            # # ======================================================
            # # Alternative model: Take action in the direction of Waypoint if reward > 0 else wait (None)
            # # ------------------------------------------------------
            # # Check previous reward for desired action ( self.next_waypoint ) in similar state ( state )
            # if self.Q[state]['reward'] > 0:
            #     action = self.next_waypoint
            # else:
            #     action = None
            # # ======================================================

        else:
            print '=> State NOT found in Q-table'

            print '[SmartCab] Assigning default values for this state inside Qtable'

            self.Q[state] = {}
            self.Q[state]['count'] = 0            
            self.Q[state]['actions_rewards'] = {}
            self.Q[state]['actions_counts'] = {}
            self.Q[state]['actions_Q^'] = {}


            # action = random.choice(self.valid_actions)

            # Alternative: 
            # 'None' may be removed to motivate exploring (as was advised by udacity review): 
            action = random.choice(['forward', 'left', 'right'])

    
            # action = self.next_waypoint
            print '=> Taking random action: ' + str(action)

        # # ======================================================
        # # Alternative model: Take random action
        # # ------------------------------------------------------
        # action = random.choice(self.valid_actions)
        # print '=> Taking random action: ' + str(action)
        # # ------------------------------------------------------


        # Count the occurance of current state
        self.Q[state]['count'] += 1

        # Execute action and get reward
        print '--------------------------------------------------'
        reward = self.env.act(self, action)
        print '--------------------------------------------------'
        print '=> Reward: ' + str(reward)
        
        if reward == 12:
            self.trips_finished[self.trip_number] = 1
            self.total_finished += 1
        else:
            self.trips_finished[self.trip_number] = 0
        
        print ''
        print '[SmartCab] Calculating and recording current results to Qtable'

        # TODO: Learn policy based on state, action, reward
        if action in self.Q[state]['actions_rewards']:
            self.Q[state]['actions_rewards'][action] =  reward
            self.Q[state]['actions_counts'][action] +=  1
            
            Qh = self.Q[state]['actions_Q^'][action]
            Qh = Qh + (self.alpha * (reward + (self.gamma * (max(self.Q[state]['actions_Q^'].values()))) - Qh))

            # # # ======================================================
            # # # Alternative models: Q^ = 0.5 ; Q^ = 1 ; Q^ = reward
            # # # ------------------------------------------------------
            # Qh = 0.5
            # Qh = 1
            # Qh = reward
            # # # ------------------------------------------------------

            self.Q[state]['actions_Q^'][action] = Qh

        else:
            self.Q[state]['actions_rewards'][action] =  reward
            self.Q[state]['actions_counts'][action] =  1            
            # self.Q[state]['actions_Q^'][action] = 0.5
            # self.Q[state]['actions_Q^'][action] = 1
            # self.Q[state]['actions_Q^'][action] = reward
            # self.Q[state]['actions_Q^'][action] = 0.0
            # self.Q[state]['actions_Q^'][action] = 2.5
            # self.Q[state]['actions_Q^'][action] = 2.0
            self.Q[state]['actions_Q^'][action] = 1.5
            # self.Q[state]['actions_Q^'][action] = 1.0
            # self.Q[state]['actions_Q^'][action] = 0.5


        # Reward for driving in the direction of way point
        self.Q[state]['reward'] = reward        

        print '--------------------------------------------------'        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        
        self.total_reward += reward
        self.trip_reward += reward
        self.current_step += 1
        self.total_steps += 1

        print '-----------------self.Q[state]--------------------'

        # ap(self.Q[state])
        print self.Q[state]

        print '=================trips_results===================='    
        self.trips_steps[self.trip_number] = self.current_step
        print 'Steps: ' + str(self.trips_steps)        

        self.trips_rewards[self.trip_number] = self.trip_reward
        print 'Rewards: ' + str(self.trips_rewards)

        self.trips_timers[self.trip_number] = round(time.time() - self.trip_begin_time, 2)
        print 'Timers: ' + str(self.trips_timers)
        
        print 'Finished' + str(self.trips_finished)
        # print 'Timers: ' + str(self.trips_timers)

        self.results_log = 'Total steps: ' + str(self.total_steps) + '; reward: ' + str(self.total_reward) + '; finished: ' + str(self.total_finished) +  '; seconds: ' + str(round(time.time() - self.timer, 4))
        print self.results_log
        print '=================================================='

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    # e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    # sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    # sim.run(n_trials=101)  # run for a specified number of trials
    sim.run(n_trials=100)  # run for a specified number of trials
    
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
