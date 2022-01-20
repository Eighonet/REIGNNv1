#!/bin/sh
mkdir processed_data
cd processed_data
wget https://n-usr-rhikf.s3pd02.sbercloud.ru/b-usr-rhikf-6g8/SSORC_CS_2010_2021_authors_edge_list.csv
wget https://n-usr-rhikf.s3pd02.sbercloud.ru/b-usr-rhikf-6g8/SSORC_CS_2010_2021_authors_edges_papers_indices.csv
wget https://n-usr-rhikf.s3pd02.sbercloud.ru/b-usr-rhikf-6g8/SSORC_CS_2010_2021_authors_features.csv
wget https://n-usr-rhikf.s3pd02.sbercloud.ru/b-usr-rhikf-6g8/SSORC_CS_2010_2021_papers_features_vectorized_compressed_32.csv
cd ../datasets/SSORC_CS_10_21_1437_3164_unfiltered
wget https://n-usr-rhikf.s3pd02.sbercloud.ru/b-usr-rhikf-6g8/SSORC_CS_10_21_1437_3164_unfiltered_authors_edges_papers_indices.csv

