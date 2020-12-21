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
