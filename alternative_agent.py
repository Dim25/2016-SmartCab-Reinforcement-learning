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
# 

import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
        self.random_actions = {}

        self.trip_reward = 0
        self.total_reward = 0
        self.total_finished = 0
        
        self.current_step = 1
        self.total_steps = 1
        self.trip_number = 0

        # self.max_trips = 50
        # self.max_trips = 250
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

            f = open('log.txt','ab+')
            f.write(self.results_log)
            f.write("\n")
            f.close()

            print self.trips_finished.values()
            print self.trips_steps.values()
            print self.trips_rewards.values()

            x = self.trips_finished.keys()
            y = self.trips_rewards.values()
            ysize = self.trips_rewards.values()
                

            for i in range (0, len(ysize)):
                ysize[i] = ( self.trips_rewards.values()[i] / self.trips_steps.values()[i] )
                y[i] = ysize[i]
                # ysize[i] = 1.5
            print ysize

            for i in range (0, len(x)):
                # Ploting action types

                # self.random_actions[self.trip_number] = 2
                # marker_based_on_action_type = int( self.random_actions.values()[i] )
                if self.random_actions.values()[i] == 0:
                    # No exploring actions per trip
                    # marker_type = "o"
                    plt.plot(x[i], -1, linestyle="None", marker=r'$'+str(self.random_actions.values()[i])+'$', markersize=10, color="black")

                elif self.random_actions.values()[i] <= 2:
                    # 1-2 exploring actions per trip
                    plt.plot(x[i], -1, linestyle="None", marker=r'$'+str(self.random_actions.values()[i])+'$', markersize=10, color="orange")

                elif self.random_actions.values()[i] >= 3:
                    # 3+ actions of exploring per trip
                    plt.plot(x[i], -0.85, linestyle="None", marker=r'$'+str(self.random_actions.values()[i])+'$', markersize=10, color="red")
                

                print i
                
                marker_type = "o"
                if self.trips_finished.values()[i] == 1:
                    plt.plot(x[i], y[i], linestyle="None", marker=marker_type, markersize=ysize[i]*10, color="green")
                else:
                    plt.plot(x[i], y[i], linestyle="None", marker=marker_type, markersize=ysize[i]*10, color="red")
                
            plt.plot(x, y, linestyle="dotted", color="red")

            plt.grid(True)

            plt.xlim(np.min(x)-1.3, np.max(x)+1.3) #optional 
            plt.ylim(np.min(y)-1.3, np.max(y)+1.3) #optional 

            plt.xlabel("Trips")
            # plt.ylabel("Finished")
            plt.ylabel("Reward/Steps")

            # plt.legend()

            plt.text(0.5, -1.5, self.results_log, fontsize=12)
            # plt.text(0.5, 0.5,' [ matplotlib ] ',
            #          horizontalalignment='left',
            #          verticalalignment='top')
            #          #transform = plt.ax.transAxes)

            # plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$')

            plt.savefig('test.png')

            plt.show()

            # plt.plot(t, s)

            # plt.xlabel('time (s)')
            # plt.ylabel('voltage (mV)')
            # plt.title('About as simple as it gets, folks')
            # plt.grid(True)
            # plt.savefig("test.png")
            # plt.show()

            # datatype=[('index',numpy.float32), ('floati',numpy.float32), 
            #         ('floatq',numpy.float32)]
            # filename='bigdata.bin'

            # data = numpy.memmap(filename, datatype, 'r') 
            # plt.plot(data['floati'],data['floatq'],'r,')
            # plt.grid(True)
            # plt.title("Signal-Diagram")
            # plt.xlabel("Sample")
            # plt.ylabel("In-Phase")
            # plt.savefig('foo2.png')

            sys.exit(0) # exit if reached the self.max_trips from config
        
        self.random_actions[self.trip_number] = 0
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
            
            # Stimulate exploring within first 10% of trips
            # if self.trip_number < (self.max_trips / 10):
            #     self.epsilon
            #     epsilon_multiplier  

            # round( self.trip_number / ( self.max_trips / 10 ),0 )

            if random_value > self.epsilon:
                self.random_actions[self.trip_number] += 1
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
            # self.random_actions[self.trip_number] = 2
            self.random_actions[self.trip_number] += 1

    
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
            self.Q[state]['actions_Q^'][action] = 1.5  # <=====
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
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=1001)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
