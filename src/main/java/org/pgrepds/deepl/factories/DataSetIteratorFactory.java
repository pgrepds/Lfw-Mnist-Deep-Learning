package org.pgrepds.deepl.factories;

import java.io.IOException;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import org.pgrepds.deepl.iterators.LfwEvalDataSetIterator;
import org.pgrepds.deepl.iterators.LfwTrainingDataSetIterator;
import org.pgrepds.deepl.iterators.MnistEvalDataSetIterator;
import org.pgrepds.deepl.iterators.MnistTrainingDataSetIterator;

/**
 * Basic factory class for {@link DataSetIterator}s. 
 * 
 * @author David Scholz
 */
@SuppressWarnings("deprecation")
public final class DataSetIteratorFactory {
	
	public DataSetIteratorFactory() {
		
	}
	
	public DataSetIterator createDataSetIterator(DataSetIteratorType type) throws IOException {
		
		DataSetIterator iterator = null;
		
		switch (type) {
			case MNIST_TRAINING: 
				iterator = new MnistTrainingDataSetIterator();
				break;
			case MNIST_EVAL:
				iterator = new MnistEvalDataSetIterator();
				break;
			case LFW_TRAINING:
				iterator = new LfwTrainingDataSetIterator();
				break;
			case LFW_EVAL:
				iterator = new LfwEvalDataSetIterator();
				break;
			// this can actually never happen, maybe due reflection magic. Better safe then sorry.
			default: 
				throw new IllegalArgumentException("Iterator type is unknown!" + type.name());
		}
		
		return iterator;
	}

	/**
	 * 
	 * Representing the iterator type.	 
	 * 
	 */
	public enum DataSetIteratorType {
		MNIST_TRAINING,
		MNIST_EVAL, 
		@Deprecated
		LFW_TRAINING,
		@Deprecated
		LFW_EVAL
	}
}
