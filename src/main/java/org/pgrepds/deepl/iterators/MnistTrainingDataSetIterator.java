package org.pgrepds.deepl.iterators;

import java.io.IOException;

import org.pgrepds.deepl.utils.MnistNetworkConstants;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;

/**
 * Purpose of this class is to have all iterator constants at one place. Makes testing much easier.
 * 
 * @author David Scholz
 */
public class MnistTrainingDataSetIterator extends MnistDataSetIterator {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 8461657372724656299L;

	public MnistTrainingDataSetIterator() throws IOException {
		super(MnistNetworkConstants.BATCH_SIZE, true, MnistNetworkConstants.RANDOM_SEED);
	}

}
