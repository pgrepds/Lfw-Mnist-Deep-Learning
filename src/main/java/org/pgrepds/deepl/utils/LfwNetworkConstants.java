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

import org.datavec.image.loader.LFWLoader;

/**
 * Basic constants for networks responsible for training with the LFW-Dataset, such as learning rate.
 * 
 * @author David Scholz
 */
public final class LfwNetworkConstants {
	
	/**
	 * Learning rate.
	 */
	public static final double LEARNING_RATE = 0.006;
	
	/**
	 * Epoch batch size.
	 */
	public static final int BATCH_SIZE = 200;  

	/**
	 * The column size.
	 */
	public static final int COLUMN_COUNT = 40;

	/**
	 * The row size.
	 */
	public static final int ROW_COUNT = 40;

	/**
	 * The random seed for the weight initialization.
	 */
	public static final int RANDOM_SEED = 42;

	/**
	 * Number of samples per iteration.
	 */
	public static final int SAMPLE_COUNT = 1000;
	
	/**
	 * Depth of the matrix.
	 */
	public static final int DEPTH = 3;
	
	/**
	 * Output classes.
	 */
	public static final int OUTPUT_CLASS_COUNT = LFWLoader.NUM_LABELS;
	
	private LfwNetworkConstants() {
		
	}

}
