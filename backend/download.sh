#!/usr/bin/env bash

mkdir data || return
cd data || exit
wget -nc https://n-usr-rhikf.s3pd02.sbercloud.ru/b-usr-rhikf-6g8/SSORC_CS_2010_2021_authors_edge_list.pkl
wget -nc https://n-usr-rhikf.s3pd02.sbercloud.ru/b-usr-rhikf-6g8/SSORC_CS_2010_2021_authors_edges_papers_indices.pkl
wget -nc https://n-usr-rhikf.s3pd02.sbercloud.ru/b-usr-rhikf-6g8/SSORC_CS_2010_2021_authors_features.pkl
wget -nc https://n-usr-rhikf.s3pd02.sbercloud.ru/b-usr-rhikf-6g8/SSORC_CS_2010_2021_papers_features_vectorized_compressed_32.pkl
wget -nc https://n-usr-rhikf.s3pd02.sbercloud.ru/b-usr-rhikf-6g8/SSORC_CS_2010_2021_papers_edge_list_indexed.pkl
wget -nc https://n-usr-rhikf.s3pd02.sbercloud.ru/b-usr-rhikf-6g8/author_2_id.pkl
wget -nc https://n-usr-rhikf.s3pd02.sbercloud.ru/b-usr-rhikf-6g8/REIGNN_test_v4.pt
wget -nc https://n-usr-rhikf.s3pd02.sbercloud.ru/b-usr-rhikf-6g8/author_data.json