import os

# List of file names
file_names = [
    "0-create_database_if_missing.sql",
    "1-first_table.sql",
    "2-list_values.sql",
    "3-insert_value.sql",
    "4-best_score.sql",
    "5-average.sql",
    "6-avg_temperatures.sql",
    "7-max_state.sql",
    "8-genre_id_by_show.sql",
    "9-no_genre.sql",
    "10-count_shows_by_genre.sql",
    "11-rating_shows.sql",
    "12-rating_genres.sql",
    "13-uniq_users.sql",
    "14-country_users.sql",
    "15-fans.sql",
    "16-glam_rock.sql",
    "17-store.sql",
    "18-valid_email.sql",
    "19-bonus.sql",
    "20-average_score.sql",
    "21-div.sql",
    "22-list_databases",
    "23-use_or_create_database",
    "24-insert",
    "25-all",
    "26-match",
    "27-count",
    "28-update",
    "29-delete",
    "30-all.py",
    "31-insert_school.py",
    "32-update_topics.py",
    "33-schools_by_topic.py",
    "34-log_stats.py"
]

# Base directory where files will be created
base_directory = "/home/j0hn/Downloads/alu-machine_learning-main/pipeline/databases"

# Create empty files
for file_name in file_names:
    file_path = os.path.join(base_directory, file_name)
    with open(file_path, 'w') as f:
        pass
    print(f"Created file: {file_path}")

