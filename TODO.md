1. A single run should consitute and contain all responses in a single directory, all subjects in a single folder with a single file per subject inside the `responses` dir. Do not replicate the persona data into the results files (other than the subject id) and run 
3. Try to get logprobs all the time, faily with warning and set logporbs as null if not possible
4. We need a function to nicely extract the responses from the responses.jsonl data into tabular format that maintains the most important data.
5. expect .env file with endpoint and api key, add to .gitignore load at runtime
6. ensure that we set the seed for each call and save it into output data
7. How can be allow user to input own config - I want to select only certain AffluenceLevel's and Countries with probabilities of being selected