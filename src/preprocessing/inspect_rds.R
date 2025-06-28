#!/usr/bin/env Rscript

# Script to inspect and convert RDS files to CSV
# Usage: Rscript inspect_rds.R

# Function to inspect and convert RDS file
inspect_and_convert_rds <- function(file_path) {
  cat("Inspecting file:", file_path, "\n")

  tryCatch({
    # Try to read the RDS file
    data <- readRDS(file_path)

    # Print information about the object
    cat("Object class:", class(data), "\n")
    cat("Object type:", typeof(data), "\n")
    cat("Object length:", length(data), "\n")

    if (is.data.frame(data)) {
      cat("Data frame dimensions:", dim(data), "\n")
      cat("Column names:", paste(names(data), collapse = ", "), "\n")
      cat("First few rows:\n")
      print(head(data))

      # Convert to CSV
      output_file <- gsub("\\.rds$", ".csv", file_path)
      write.csv(data, output_file, row.names = FALSE)
      cat("Successfully converted to:", output_file, "\n")

    } else if (is.list(data)) {
      cat("List with", length(data), "elements\n")
      cat("List names:", paste(names(data), collapse = ", "), "\n")

      # If it's a list, try to find data frames
      for (i in seq_along(data)) {
        if (is.data.frame(data[[i]])) {
          cat("Found data frame in element", i, ":", names(data)[i], "\n")
          cat("Dimensions:", dim(data[[i]]), "\n")
          cat("Columns:", paste(names(data[[i]]), collapse = ", "), "\n")

          # Convert this data frame to CSV
          output_file <- gsub("\\.rds$", paste0("_", names(data)[i], ".csv"), file_path)
          write.csv(data[[i]], output_file, row.names = FALSE)
          cat("Converted to:", output_file, "\n")
        }
      }

    } else {
      cat("Object is not a data frame or list. Cannot convert to CSV.\n")
      cat("Object structure:\n")
      str(data)
    }

  }, error = function(e) {
    cat("Error reading RDS file:", e$message, "\n")
  })
}

# Main execution
main <- function() {
  # File to inspect
  file_path <- "data/raw/Pollinator_taxonomy.rds"

  if (file.exists(file_path)) {
    inspect_and_convert_rds(file_path)
  } else {
    cat("File not found:", file_path, "\n")
  }
}

# Run the script
main()