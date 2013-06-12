/***
  *  Project:
  *
  *  File: config_file_reader.hpp
  *  Created: Jul 11, 2012
  *  Modified: Tue 11 Jun 2013 11:32:45 AM PDT
  *
  *  Author: Abhinav Sarje <asarje@lbl.gov>
  */

#ifndef __CONFIG_FILE_READER__
#define __CONFIG_FILE_READER__

#include <fstream>

#include "typedefs.hpp"


namespace hir {

	// this is a singleton stateless class
	class ConfigFileReader {
		private:
			ConfigFileReader() { }
			ConfigFileReader(const ConfigFileReader&);
			ConfigFileReader& operator=(const ConfigFileReader&);

		public:
			static ConfigFileReader& instance() {
				static ConfigFileReader config_file_reader;
				return config_file_reader;
			} // instance()

	}; // class ConfigFileReader

} // namespace hir


#endif /* __HIG_FILE_READER__ */
