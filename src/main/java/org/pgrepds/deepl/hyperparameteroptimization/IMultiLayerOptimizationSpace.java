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
