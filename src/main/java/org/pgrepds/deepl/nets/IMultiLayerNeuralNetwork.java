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

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * Basic API for a multilayered neural network to give so convenient methods.
 * 
 * @author David Scholz
 */
public interface IMultiLayerNeuralNetwork {
	
	/**
	 * Trains the network with the underlying model.
	 */
	void trainNetwork();
	
	/**
	 * Evaluates the model. Results are written to an log file. If no logging framework is defined 
	 * the result is written to stdout (very slow).
	 */
	void evalModel();
	
	/**
	 * 
	 * Builds the networks model.
	 * 
	 * @param config the {@link IMultiLayerConfiguration} of the model.
	 */
	void buildModel(IMultiLayerConfiguration config);
	
	/**
	 * 
	 * @return the {@link IMultiLayerConfiguration} of the underlying model.
	 */
	IMultiLayerConfiguration getMultiLayerConfiguration();
	
	/**
	 * 
	 * @return the {@link MultiLayerNetwork}, which represents the underlying model.
	 */
	MultiLayerNetwork getMultiLayerModel();

}
