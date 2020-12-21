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
package org.pgrepds.deepl.hyperparameteroptimization;

import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * @author David Scholz
 */
public interface IMultiLayerHyperparameterOptimizationNetwork {
	
	/**
	 * Builds the model to optimize using {@link ParameterSpace}s, e.g. for the learning rate and the layer depth.
	 * 
	 * @param space the {@link IMultiLayerOptimizationSpace}.
	 */
	void build(IMultiLayerOptimizationSpace space);
	
	/**
	 * Starts the optimization.
	 */
	void startOptimization();
	
	/**
	 * Some basics statistics.
	 * 
	 * @return the {@link OptimizationResult} of the hyperparameter optimization.
	 */
	OptimizationResult getOptimizationResult();
	
	/**
	 * @return the local optimal {@link MultiLayerNetwork}.
	 */
	MultiLayerNetwork getOptimizedMultiLayerNetworkModel();

}
