package org.pgrepds.deepl.utils;

/**
 * Basic utility class. The main purpose of this class was to determine configuration problems with platform 
 * specific libraries, which prevented deeplearning4j to work.
 * 
 * @author David Scholz
 */
public final class MnistNetworkUtils {
	
	
	public MnistNetworkUtils() {
		
	}
	
	private static void load(String name) {
	  
		try{
    	  
          System.out.println("Trying to load: "+name);
          System.loadLibrary(name);
          
		}catch (Throwable e){
          System.out.println("Failed: "+e.getMessage());
          return;
		}
      
      System.out.println("Success");
	}

	public static void testNativeLibFailures() {
      load("libwinpthread-1");
      load("libstdc++-6");
      load("libquadmath-0");
      load("libopenblas");
      load("libgomp-1");
      load("libgfortran-3");
      load("libgcc_s_seh-1");
	}

}
