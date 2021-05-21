
# Commit policy

Commits in english
A good commit should always be able to complete the following sentence. 
* if applied, this commit will [commit message]*

## Practical example:

```git
fix(screen): bug in the customer model
Feat(model): Add import project functionality
```
## commit tags
* fix(scope)
* feat(scope)
* wip(scope)
* refat(scope)
* del(scope)

* Use **imperative** verbal mode!


***

# Branches Style

Branches in English.
Branches have a *TAG* that references the issue that this branch will resolve.

* Practical example:

A branch for the issue "#4 use significant variables names" should look like:

```git
git checkout -b 4-use-significant-variables-names
```

* If the issue title is way too big, use only the first significant words to create it, do not create text-sized branches. 

***

# Git Workflow

![Master-Devel](https://user-images.githubusercontent.com/18370133/54881036-5dc07e00-4e2a-11e9-8164-4ee64526939b.png)

## Develop and Master Branches
Instead of a single master branch, this workflow uses two branches to record the history of the project. The `master` branch stores the official release history, and the `develop` branch serves as an integration branch for features. It's also convenient to tag all commits in the `master` branch with a version number.

The first step is to complement the default `master` with a `develop` branch. A simple way to do this is for one developer to create an empty `develop` branch locally and push it to the server:

## Feature Branches

Each new feature should reside in its own branch, which can be pushed to the central repository for backup/collaboration. But, instead of branching off of `master`, feature branches use `develop` as their parent branch. When a feature is complete, it gets merged back into `develop`. Features should **never** interact directly with `master`.

![Feature Branches](https://user-images.githubusercontent.com/18370133/54881065-97918480-4e2a-11e9-8560-13c1336c05d6.png)

## Creating a feature branch

```bash
git checkout develop
git checkout -b feature_branch
```

## Finishing a feature branch

When you’re done with the development work on the feature, the next step is to merge the `feature_branch` into `develop`. You should then open a **Pull Request** from `feature_branch` into `develop`, which will be reviewed by one of the EPS team members.

## Release Branches

![Release Branches](https://user-images.githubusercontent.com/18370133/54881189-26eb6780-4e2c-11e9-98ee-d4e203e0c479.png)

Once `develop` has acquired enough features for a release (or a predetermined release date is approaching), you fork a `release` branch off of `develop`. Creating this branch starts the next release cycle, so no new features can be added after this point—only bug fixes, documentation generation, and other release-oriented tasks should go in this branch. Once it's ready to ship, the `release` branch gets merged into master and tagged with a version number. In addition, it should be merged back into `develop`, which may have progressed since the release was initiated.

Using a dedicated branch to prepare releases makes it possible for one team to polish the current release while another team continues working on features for the next release. It also creates well-defined phases of development (e.g., it's easy to say, “This week we're preparing for version 4.0,” and to actually see it in the structure of the repository).

Making `release` branches is another straightforward branching operation. Like feature branches, `release` branches are based on the `develop` branch. A new `release` branch can be created using the following methods.

```bash
git checkout develop
git checkout -b release/0.1.0
```

Once the release is ready to ship, it will get merged it into master and develop, then the `release` branch will be deleted. It’s important to merge back into develop because critical updates may have been added to the `release` branch and they need to be accessible to new features. If your organization stresses code review, this would be an ideal place for a pull request.



***

# Python Style

The style guide used is Python's [PEP 8](https://www.python.org/dev/peps/pep-0008/)

* **snake_case**, NOT camelCase

Use:
```Python
class Person(models.Model):
    first_name = models.CharField(max_length=20)
    last_name = models.CharField(max_length=40)
```

Do **NOT** do the following:
```Python
class Person(models.Model):
    FirstName = models.CharField(max_length=20)
    Last_Name = models.CharField(max_length=40)
```

* 4 space indent

* 4 space indent for functions with big parameters

Use:
```Python
raise AttributeError(
    'Here is a multine error message '
    'shortened for clarity.'
)
```

Do **NOT** use:
```Python
raise AttributeError('Here is a multine error message '
                     'shortened for clarity.')
```

* Single quote for strings

* Function docstrings follow the [PEP 257](https://www.python.org/dev/peps/pep-0257/)

```Python
def test_foo():
    """
    A test docstring looks like this (#123456).
    """
    ...
```

* **Meta** class must be implemented just after class attributes and must be separated with just ONE line.

```Python
class Person(models.Model):
    first_name = models.CharField(max_length=20)
    last_name = models.CharField(max_length=40)

    class Meta:
        verbose_name_plural = 'people'
```

***

# References

[1] https://br.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow

[2] https://br.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow

[How to write a git commit message](https://chris.beams.io/posts/git-commit/)
