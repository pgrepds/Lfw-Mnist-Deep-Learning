package org.pgrepds.deepl.factories;

/**
 * Basic {@link Exception} for {@link IInternalDataSetIteratorFactory} specific exceptions.
 * 
 * @author David Scholz
 */
public class DataSetIteratorFactoryException extends Exception {

	private static final long serialVersionUID = -5727523879829859849L;
	
	public DataSetIteratorFactoryException(final String msg) {
		super(msg);
	}
	
	public DataSetIteratorFactoryException(final String msg, Exception e) {
		super(msg, e);
	}

}
