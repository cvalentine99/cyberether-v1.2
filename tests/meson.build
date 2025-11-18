if catch2_dep.found()
    subdir('memory')
    subdir('modules')
    subdir('remote')
endif

test('memory', executable(
    'jetstream-memory', 'memory.cc',
    dependencies: libjetstream_dep,
), is_parallel: false, timeout: 0)

# Component tests
subdir('components')
