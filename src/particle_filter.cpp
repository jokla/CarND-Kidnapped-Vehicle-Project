/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <math.h>


#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;

	default_random_engine gen;

	// This line creates a normal (Gaussian) distribution for x
	normal_distribution<double> dist_x(x, std[0]);

	// Create normal distributions for y and theta
	normal_distribution<double> dist_y(y,  std[1]);
	normal_distribution<double> dist_theta(theta,  std[2]);

	weights.resize(num_particles);


	for (int i = 0; i < num_particles; i++) {
			Particle p_temp;
			//   Sample  and from these normal distrubtions like this:
			//	 sample_x = dist_x(gen);
			//	 where "gen" is the random engine initialized earlier.
			p_temp.id = i;
			p_temp.x = dist_x(gen);
			p_temp.y = dist_y(gen);
			p_temp.theta = dist_theta(gen);
			p_temp.weight = 1.0;
			weights[i] = 1.0;

			particles.push_back(p_temp);

		}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/


	default_random_engine gen;
	// This line creates a normal (Gaussian) distribution for x
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0,  std_pos[1]);
	normal_distribution<double> dist_theta(0,  std_pos[2]);

	for (unsigned int i = 0; i < num_particles; i++) {

			double x = particles[i].x;
			double y = particles[i].y;
			double theta = particles[i].theta;

			if(fabs(yaw_rate) > 0.0001){
					x = particles[i].x + velocity/yaw_rate*((sin(particles[i].theta + yaw_rate*delta_t)) - sin(particles[i].theta));
					y = particles[i].y + velocity/yaw_rate*((- cos(particles[i].theta + yaw_rate*delta_t)) + cos(particles[i].theta));
					theta = particles[i].theta + yaw_rate * delta_t;
				}
			else
				{
					x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
					y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
				}


			particles[i].x = x + dist_x(gen);
			particles[i].y = y + dist_y(gen);
			particles[i].theta = theta + dist_theta(gen);
		}

}


double compute_dist(const LandmarkObs &predicted, const LandmarkObs &observation)
{
	return std::sqrt(std::pow(observation.x - predicted.x,2) + std::pow(observation.y - predicted.y,2) );
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for (unsigned int i = 0; i < observations.size(); i++)
		{

			double dist_min = 10e10;
			int id_min = -1;

			for (unsigned int j = 0; j < predicted.size(); j++)
				{
					double dist = compute_dist(observations[i],  predicted[j] );

					if (dist < dist_min)
						{
							dist_min = dist;
							id_min = j;//predicted[j].id;
						}
				}

			observations[i].id =	id_min; // Use index
		}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
																	 const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
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

	// Observations expressed in map coordinate system
	weights.clear();

	for (unsigned int i = 0; i < num_particles; ++i) {
			std::vector<LandmarkObs> m_observations(observations.size());

			for (unsigned int j = 0; j < m_observations.size(); j++)
				{
					m_observations[j].x = particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
					m_observations[j].y = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);
					m_observations[j].id = observations[j].id;
				}

			std::vector<LandmarkObs> predicted;

			for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
					LandmarkObs p;
					p.id = map_landmarks.landmark_list[j].id_i;
					p.x = map_landmarks.landmark_list[j].x_f;
					p.y = map_landmarks.landmark_list[j].y_f;

					double dist = std::sqrt(std::pow(p.x - particles[i].x,2) + std::pow(p.y - particles[i].y,2) );

					if (dist <= sensor_range )
						predicted.push_back(p);
				}

			dataAssociation(predicted, m_observations);

			particles[i].weight = 1.0;

			for (unsigned int j = 0; j < m_observations.size(); j++) {

					// calculate normalization term
					double gauss_norm= (1.0/(2 * M_PI * std_landmark[0] * std_landmark[1]));

					int id = m_observations[j].id;
					double exponent = (pow((m_observations[j].x - predicted[id].x),2.0))/(2 * std_landmark[0]*std_landmark[0]) +
							(pow((m_observations[j].y - predicted[id].y),2.0))/(2 * std_landmark[1]*std_landmark[1]);

					// calculate weight using normalization terms and exponent
					double weight = gauss_norm * exp(-exponent);
					particles[i].weight = particles[i].weight * weight;
				}

			weights.push_back(particles[i].weight);
		}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::discrete_distribution<int> d(weights.begin(), weights.end());
	std::vector<Particle> new_particles;
	default_random_engine gen;
	for (unsigned i = 0; i < num_particles; i++) {
			auto ind = d(gen);
			new_particles.push_back(std::move(particles[ind]));
		}
	particles = std::move(new_particles);
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
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
