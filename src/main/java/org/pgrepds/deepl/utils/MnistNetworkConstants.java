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
package org.pgrepds.deepl.utils;

/**
 * Basic constants for networks responsible for training with the MNIST-Dataset, such as learning rate.
 * 
 * @author David Scholz
 */
public final class MnistNetworkConstants {
	
	/**
	 * Number of rows of one input picture. (must be known according to the dataset, e.g. 28x28 pixels) 
	 */
	public static final int ROW_COUNT = 28;
	
	/**
	 * Number of columns of one input picture. (must be known according to the dataset, e.g. 28x28 pixels)
	 */
	public static final int COLUMN_COUNT = 28;
	
	/**
	 * Number of output classes.
	 */
	public static final	int OUTPUT_CLASSES_COUNT = 10;
	
	/**
	 * Epoch batch size.
	 */
	public static final int BATCH_SIZE = 128;
	
	/**
	 * Random seed for reproducibility.
	 */
	public static final int RANDOM_SEED = 42;
	
	/**
	 * Number of epochs to perform.
	 */
	public static final int EPOCH_COUNT = 15;
	
	/**
	 * Learning rate.
	 */
	public static final double LEARNING_RATE = 0.006;
	
	/**
	 * The depth of the MNIST dataset.
	 */
	public static final int DEPTH = 1;
	
	/**
	 * Lower bound of the learning rate hyperparameter space.
	 */
	public static final double HYPERPARAMTER_LEARNINGRATE_LOWER_BOUND = 0.0001;
	
	/**
	 * Upper bound of the learning rate hyperparameter space.
	 */
	public static final double HYPERPARAMTER_LEARNINGRATE_UPPER_BOUND = 0.1;
	
	/**
	 * Lower bound of the random distribution of the layer depth hyperparameter space.
	 */
	public static final int HYPERPARAMTER_LAYERDEPTH_LOWER_BOUND = 16;
	
	/**
	 * Upper bound of the random distribution of the layer depth hyperparameter space.
	 */
	public static final int HYPERPARAMTER_LAYERDEPTH_UPPER_BOUND = 256;
	
	private MnistNetworkConstants() {
		
	}

}
