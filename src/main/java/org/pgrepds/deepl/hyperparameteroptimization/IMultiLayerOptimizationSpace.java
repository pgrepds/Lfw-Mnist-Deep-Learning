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

import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.layers.LayerSpace;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;

/**
 * {@link MultiLayerSpace} for optimization hyperparameter, e.g. layer depth or learning rate.
 * 
 * @author David Scholz
 */
public interface IMultiLayerOptimizationSpace {
	
	/**
	 * Adding a new {@link LayerSpace} to the {@link IMultiLayerOptimizationSpace}.
	 * 
	 * @param layer the layer space to add.
	 */
	void addLayer(LayerSpace<?> layer);
	
	/**
	 * @return the configured {@link MultiLayerSpace}.
	 */
	MultiLayerSpace createMultiLayerSpace();
	
	ParameterSpace<Integer> getLayerDepthParameterSpace(); 

}
