#!/bin/bash
set -e

# Wait for MongoDB instances to be available
echo "Waiting for MongoDB instances to be available..."
sleep 10

# Initialize the replica set with retry logic
mongosh --host mongo1 --port 30001 <<EOF
  // Function to retry operations with delay
  function retryOperation(operation, maxRetries = 10, delay = 2000) {
    let retries = 0;
    let result;
    
    while (retries < maxRetries) {
      try {
        result = operation();
        if (result && result.ok === 1) {
          return result;
        }
        print("Operation returned non-ok result, retrying in " + delay + "ms... (" + (retries + 1) + "/" + maxRetries + ")");
      } catch (e) {
        print("Error occurred: " + e.message + ", retrying in " + delay + "ms... (" + (retries + 1) + "/" + maxRetries + ")");
      }
      
      // Increase delay with each retry
      sleep(delay);
      delay = Math.min(delay * 1.5, 10000); // Exponential backoff, max 10s
      retries++;
    }
    
    throw new Error("Operation failed after " + maxRetries + " retries");
  }

  // Check if replica set is already initialized
  let isInitialized = false;
  
  try {
    let status = rs.status();
    if (status && status.ok === 1) {
      print("Replica set is already initialized");
      printjson(status);
      isInitialized = true;
    }
  } catch (e) {
    // Expected error if not initialized yet
    if (e.codeName === "NotYetInitialized" || /no replset config has been received/.test(e.message)) {
      print("Replica set not yet initialized, will proceed with initialization");
    } else {
      print("Warning: Unexpected error checking replica set status: " + e.message);
    }
  }
  
  if (!isInitialized) {
    print("Initializing replica set...");
    
    // Configure and initiate the replica set
    var config = {
      _id: "my-replica-set",
      members: [
        {_id: 0, host: "mongo1:30001", priority: 2},
        {_id: 1, host: "mongo2:30002", priority: 1},
        {_id: 2, host: "mongo3:30003", priority: 1}
      ]
    };
    
    try {
      let initResult = retryOperation(() => rs.initiate(config));
      print("Replica set initialization result:");
      printjson(initResult);
      
      // Wait for replica set to stabilize
      print("Waiting for replica set to stabilize...");
      sleep(5000);
      
      // Check final status
      let finalStatus = retryOperation(() => rs.status());
      print("Final replica set status:");
      printjson(finalStatus);
    } catch (e) {
      print("Failed to initialize replica set: " + e.message);
      // Don't exit with error as our Python code can handle this situation
    }
  }
EOF

echo "MongoDB replica set initialization completed" 