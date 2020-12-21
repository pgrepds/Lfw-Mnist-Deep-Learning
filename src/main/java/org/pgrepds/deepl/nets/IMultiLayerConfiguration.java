/*
	Copyright 2020 David Scholz

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

    	http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.
 */
package org.pgrepds.deepl.nets;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.layers.convolution.ConvolutionLayer;
import org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer;
import org.deeplearning4j.nn.weights.WeightInit;

/**
 * API for to configure {@link MultiLayerConfiguration}s.
 * 
 * @author David Scholz
 */
public interface IMultiLayerConfiguration {
	
	/**
	 * 
	 * @return the {@link WeightInit} of the network.
	 */
	WeightInit getWeightInit();
	
	/**
	 * 
	 * @return the random seed.
	 */
	int getRandomSeed();
	
	/**
	 * 
	 * @return the name of the {@link Updater} which is used in the network.
	 */
	String getUpdaterName();
	
	/**
	 * 
	 * @return the learning rate of the network;
	 */
	double getLearningRate();
	
	/**
	 * 
	 * @return <code>true</> if the network is using backpropagation to adjust the weights.
	 */
	boolean isUsingBackpropagation();
	
	/**
	 * 
	 * Adds a new {@link Layer}, e.g. a {@link DenseLayer} or {@link ConvolutionLayer}.
	 * 
	 * @param pos the position of the layer.
	 * @param layer the {@link Layer} to add to the configuration.
	 */
	void addLayer(int pos, Layer layer);
	
	/**
	 * 
	 * Creates the configured {@link MultiLayerConfiguration}.
	 * 
	 * @return immutable {@link MultiLayerConfiguration}.
	 */
	MultiLayerConfiguration createMultiLayerConfiguration();

}
