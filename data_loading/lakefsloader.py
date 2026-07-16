"""
Collection of functions to interact with lakefs and the s3 storage.
"""

import os
import boto3
import logging
from pathlib import Path

from typing import Tuple, List

class LakeFSLoader():

    def __init__(
            self, 
            repo_name: str,
            branch_id: str,
            local_cache_path: str,
            endpoint: str | None,
            ca_path: str | None,
            access_key: str | None,
            secret_key: str | None,
            ):

        self.repo_name = repo_name
        self.branch_id = branch_id
        if not self.branch_id.endswith('/'):
            self.branch_id += '/'
        self.local_cache_path = local_cache_path
        
        if self.local_cache_path is None:
            raise ValueError(f"local_cache_path is None. Please check your LakeFS config file (expected key: 'cache'). Loaded config keys: {list(locals().keys())}")

        if not self.local_cache_path.endswith('/'):
            self.local_cache_path += '/'

        # create a s3 client from keys defined in config if defined
        # create a s3 client from keys defined in config if defined
        if endpoint and access_key and secret_key:
            verify_ssl = ca_path if ca_path else False
            self.lakefs = boto3.client(
                's3',
                endpoint_url=endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                verify=verify_ssl,
            )
        # create the client from the default ~/.aws/config and ~/.aws/credentials file
        else:
            self.lakefs = boto3.client('s3')
    
        
    def check_num_missing_files(self, object_names: List):
        """
        Check the objects if they are already in the cache, returns the number which still need to be downloaded.

        Return:
            (int): Number of files that still need to be downloaded.
        """
        not_in_cache = 0
        for obj_name in object_names:
            file_path = Path(self.local_cache_path) / obj_name
            if not os.path.exists(file_path):
                not_in_cache += 1

        return not_in_cache
    

    def get_local_and_obj_names(self, name: str) -> Tuple:
        """
        Returns both the full local cache path and the object name (s3 path starting with the branch id),
        with which it can be identified within LakeFS.

        Args:
            name (string): Either the object name or local name.
        Return:
            Tuple(string, string): (local name, object name)
        """
        if name.startswith(self.branch_id):
            obj_name = name
        else:
            branch_key = self.branch_id.rstrip('/')
            
            # If the name contains the branch name (e.g. inside an absolute path)
            if f"/{branch_key}/" in name or name.startswith(f"{branch_key}/"):
                idx = name.find(branch_key)
                stripped = name[idx:]
            else:
                # Strip cache prefix if present (handles absolute local paths)
                stripped = name.replace(self.local_cache_path, '').lstrip('/')
                
            # Only prepend branch if the stripped name doesn't already start with it
            if stripped.startswith(self.branch_id) or stripped.startswith(branch_key):
                obj_name = stripped
            else:
                obj_name = f"{self.branch_id}{stripped}"
        
        local_name = str(Path(self.local_cache_path) / obj_name)
        return local_name, obj_name
    

    def get_branch_dir(self):
        """
        Gets the directory to the cached files

        Return:
            (pathlib.Path): The directory path of the cache corresponding to the lakefs branch id.
        """
        return Path(self.local_cache_path) / self.branch_id
    

    def check_file(self, file_name: str):
        """
        Checks if a file exists, if not downloads it.
        Args:
            file_name (string): Either the local or s3 file name
        """
        local_name, obj_name = self.get_local_and_obj_names(file_name)
        if not os.path.exists(local_name):
            os.makedirs(Path(local_name).parent, exist_ok=True)
            logging.info(f"{obj_name} missing in the cache - downloading now from s3 storage.")
            temp_name = local_name + ".tmp"
            try:
                self.lakefs.download_file(self.repo_name, obj_name, temp_name)
                os.replace(temp_name, local_name)
            except Exception as e:
                if os.path.exists(temp_name):
                    try:
                        os.remove(temp_name)
                    except Exception:
                        pass
                raise e


    def check_dir(self, dir_name: str):
        """
        Checks if a local file directory exists, if not downloads it and its subfiles.
        Args:
            local_name (string): The object name (s3 path starting with the branch id).
        """
        local_dir_name, obj_dir_name = self.get_local_and_obj_names(dir_name)
        if not os.path.exists(local_dir_name):
            os.makedirs(Path(local_dir_name), exist_ok=True)
            # search for all matching objects in lakefs
            dir_objects = self.read_s3_objects(prefix=obj_dir_name.replace(self.branch_id, ''))

            # download
            for dir_object in dir_objects:
                self.check_file(dir_object)

    # In your lakefsloader.py file, update the read_s3_objects method:

    def read_s3_objects(self, filter: str = "", prefix: str = "") -> list:
        """
        List objects in the bucket with given prefix, optionally filtered by string.
        Returns list of object keys.
        """
        objects = []

        try:
            paginator = self.lakefs.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.repo_name, Prefix=prefix)

            for page in pages:
                # Check if 'Contents' key exists in response
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        if filter in key:
                            objects.append(key)
                # Some responses might use different key names or be empty
                elif "CommonPrefixes" in page:
                    # Handle directory listings
                    for common_prefix in page["CommonPrefixes"]:
                        prefix_key = common_prefix["Prefix"]
                        if filter in prefix_key:
                            objects.append(prefix_key)

        except Exception as e:
            logging.error(f"Error listing objects with prefix '{prefix}': {e}")

        return objects


    def download_file(self, repo: str, branch: str, path: str, local_path: str):
        """
        Download a specific file from LakeFS.
        
        Args:
            repo: Repository name
            branch: Branch name
            path: Path to file in LakeFS (relative to branch root) OR absolute local path
            local_path: Local destination path
        """
        print(f"[DEBUG] download_file called with:")
        print(f"  repo: {repo}")
        print(f"  branch: {branch}")
        print(f"  path: {path}")
        print(f"  local_path: {local_path}")
        print(f"  self.local_cache_path: {self.local_cache_path}")

        # Determine the object key
        # If path is absolute and starts with local_cache_path, strip it
        if path.startswith(self.local_cache_path):
            clean_path = path[len(self.local_cache_path):].lstrip('/')
            print(f"[DEBUG] Path starts with cache path. Strip result: {clean_path}")
        else:
            # Fallback: assume it's already relative or clean it
            clean_path = path.lstrip('/')
            print(f"[DEBUG] Path does NOT start with cache path. Result: {clean_path}")

        # If the path already includes the branch (e.g. main/path/to/file), use it as is
        # Otherwise prepend branch
        if clean_path.startswith(branch):
            obj_key = clean_path
        else:
            obj_key = f"{branch.rstrip('/')}/{clean_path}"
        
        print(f"[DEBUG] Final obj_key: {obj_key}")
            
        logging.info(f"Helper: Downloading {obj_key} to {local_path}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        temp_local_path = local_path + ".tmp"
        try:
            self.lakefs.download_file(
                Bucket=repo,
                Key=obj_key,
                Filename=temp_local_path
            )
            os.replace(temp_local_path, local_path)
            print(f"[DEBUG] Download successful for {obj_key}")
            return True
        except Exception as e:
            if os.path.exists(temp_local_path):
                try:
                    os.remove(temp_local_path)
                except Exception:
                    pass
            logging.error(f"Error downloading {obj_key}: {e}")
            print(f"[DEBUG] Download FAILED for {obj_key}: {e}")
            raise e


