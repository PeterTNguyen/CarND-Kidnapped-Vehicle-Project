/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

#define NUM_PARTICLES 100
int ParticleFilter::part_id = 0;

void ParticleFilter::init(double x, double y, double theta, double std[]) 
{
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = NUM_PARTICLES;
    
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_t(theta, std[2]);

    for(int i =0 ; i < num_particles; i++)
    {
        // Create new particle
        Particle tmp;
        tmp.id = ++part_id;
        tmp.x = dist_x(gen);
        tmp.y = dist_y(gen);
        tmp.theta = dist_t(gen);
        tmp.weight = 1.0;

        particles.push_back(tmp);
    }
    
    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) 
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    for(vector<Particle>::iterator particle = particles.begin(); particle != particles.end(); ++particle)
    {
        double x, y, theta;
        x = particle->x + velocity/yaw_rate*(sin(particle->theta + yaw_rate*delta_t) - sin(particle->theta));
        y = particle->y + velocity/yaw_rate*(cos(particle->theta) - cos(particle->theta + yaw_rate*delta_t)); 
        theta = particle->theta + yaw_rate*delta_t;

        normal_distribution<double> dist_x(x, std_pos[0]);
        normal_distribution<double> dist_y(y, std_pos[1]);
        normal_distribution<double> dist_t(theta, std_pos[2]);

        particle->x = dist_x(gen);
        particle->y = dist_y(gen);
        particle->theta = dist_t(gen);
    }

}

//void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
void ParticleFilter::dataAssociation(Map map_landmarks, std::vector<LandmarkObs>& observations) 
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    //double for loop
    //Loop over Observations
    double min_dist, dist;
    for(vector<LandmarkObs>::iterator obs = observations.begin(); obs != observations.end(); ++obs)
    {

        min_dist = numeric_limits<double>::infinity();

		// Loop over all landmarks to find closest one to transformed observation
        for(vector<Map::single_landmark_s>::iterator landmark = map_landmarks.landmark_list.begin();
            landmark != map_landmarks.landmark_list.end(); ++landmark)
        {
            dist = (obs->x - landmark->x_f)*(obs->x - landmark->x_f) 
                 + (obs->y - landmark->y_f)*(obs->y - landmark->y_f);

            if(min_dist > dist)
            {
                obs->id = landmark->id_i;
				min_dist = dist;
            }

        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) 
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


	//Reset weights/max weight
	max_weight = -numeric_limits<double>::infinity();
	weights.clear();

	double weight;
	double sigx = std_landmark[0], sigy = std_landmark[1];
    for(vector<Particle>::iterator particle = particles.begin(); particle != particles.end(); ++particle)
    {
        vector<LandmarkObs> mapObss;
        LandmarkObs mapObs;
        //Transform landmark observations to map coordinates using particle's x,y,theta
        for(vector<LandmarkObs>::const_iterator obs = observations.begin(); obs != observations.end(); ++obs)
        {
            mapObs.x = particle->x + cos(particle->theta) * obs->x - sin(particle->theta) * obs->y;
            mapObs.y = particle->y + sin(particle->theta) * obs->x + cos(particle->theta) * obs->y;
            mapObss.push_back(mapObs);

        }

        //Data Association(map landmark locations, transformed observations)
        dataAssociation(map_landmarks, mapObss);

		//Update particle's weight
		if(!mapObss.empty())
		{
			weight = 1.0;
		}
		else
		{
			weight = 0.0;
		}
		for(vector<LandmarkObs>::iterator obs = mapObss.begin(); obs != mapObss.end(); ++obs)
		{
			double ux, uy, x, y, diffx, diffy, multi_prob;
			x = obs->x;
			y = obs->y;
			ux = map_landmarks.landmark_list[obs->id-1].x_f;
			uy = map_landmarks.landmark_list[obs->id-1].y_f;
			diffx = x - ux;
			diffy = y - uy;

			multi_prob = exp(-( diffx*diffx/(sigx*sigx*2.0) + diffy*diffy/(sigy*sigy*2.0) )) / (2*M_PI*sigx*sigy); 
			weight *= multi_prob;
		}
		particle->weight = weight;
		weights.push_back(weight);
		if(max_weight < weight)
			max_weight = weight;
    }

}

void ParticleFilter::resample() 
{
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution


	// New particle vector
	vector<Particle> new_particles;

	double beta = 0.0;

	// Generate random index
	uniform_int_distribution<int> rand_ind(0, num_particles-1);
	int ind = rand_ind(gen);

	// Uniform random generator [0, 2*max_weight]
	uniform_real_distribution<double> rand_weight(0.0, 2.0*max_weight);

	for(int i = 0; i < num_particles; i++)
	{
		beta += rand_weight( gen );
		while(weights[ind] < beta)
		{
			beta -= weights[ind];
			ind = (ind+1) % num_particles;
		}
		new_particles.push_back(particles[ind]);
	}


	particles = new_particles;


}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
