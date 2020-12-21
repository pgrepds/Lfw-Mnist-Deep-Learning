package org.pgrepds.deepl.iterators;

import java.io.IOException;
import java.util.Random;

import org.pgrepds.deepl.utils.LfwNetworkConstants;
import org.datavec.image.loader.LFWLoader;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;

/**
 * Purpose of this class is to have all iterator constants at one place. Makes testing much easier.
 * 
 * @author David Scholz
 */
@Deprecated
public class LfwEvalDataSetIterator extends LFWDataSetIterator {

	private static final long serialVersionUID = 6920474034392410260L;

	public LfwEvalDataSetIterator() throws IOException {
		super(
				LfwNetworkConstants.BATCH_SIZE,
				LfwNetworkConstants.SAMPLE_COUNT, 
				new int[] {LfwNetworkConstants.ROW_COUNT, LfwNetworkConstants.COLUMN_COUNT, LfwNetworkConstants.DEPTH}, 
				LFWLoader.NUM_LABELS, 
				true,
				false, 
				0.8, 
				new Random(LfwNetworkConstants.RANDOM_SEED)
	    );
		
	}

}
