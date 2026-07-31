include_guard(GLOBAL)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

get_filename_component(REFLECTGEN_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(REFLECTGEN_ROOT "${REFLECTGEN_CMAKE_DIR}/.." ABSOLUTE)

function(_reflectgen_collect_target_sources out_var source_target)
    set(result)

    get_target_property(target_sources "${source_target}" SOURCES)
    if(target_sources)
        list(APPEND result ${target_sources})
    endif()

    get_target_property(interface_sources "${source_target}" INTERFACE_SOURCES)
    if(interface_sources)
        list(APPEND result ${interface_sources})
    endif()

    get_target_property(target_source_dir "${source_target}" SOURCE_DIR)
    if(NOT target_source_dir)
        set(target_source_dir "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    set(resolved)
    foreach(item IN LISTS result)
        if(item MATCHES "\\$<")
            continue()
        endif()

        if(NOT IS_ABSOLUTE "${item}")
            set(item "${target_source_dir}/${item}")
        endif()

        get_filename_component(ext "${item}" EXT)
        string(TOLOWER "${ext}" ext)
        if(ext IN_LIST _reflectgen_source_exts)
            list(APPEND resolved "${item}")
        endif()
    endforeach()

    if(resolved)
        list(REMOVE_DUPLICATES resolved)
    endif()

    set(${out_var} ${resolved} PARENT_SCOPE)
endfunction()

function(_reflectgen_collect_target_usage out_includes out_defines out_options source_target)
    set(includes)
    set(defines)
    set(options)

    foreach(prop INCLUDE_DIRECTORIES INTERFACE_INCLUDE_DIRECTORIES)
        get_target_property(values "${source_target}" "${prop}")
        if(values)
            list(APPEND includes ${values})
        endif()
    endforeach()

    foreach(prop COMPILE_DEFINITIONS INTERFACE_COMPILE_DEFINITIONS)
        get_target_property(values "${source_target}" "${prop}")
        if(values)
            list(APPEND defines ${values})
        endif()
    endforeach()

    foreach(prop COMPILE_OPTIONS INTERFACE_COMPILE_OPTIONS)
        get_target_property(values "${source_target}" "${prop}")
        if(values)
            list(APPEND options ${values})
        endif()
    endforeach()

    foreach(list_name includes defines options)
        if(${list_name})
            list(REMOVE_DUPLICATES ${list_name})
        endif()
    endforeach()

    set(${out_includes} ${includes} PARENT_SCOPE)
    set(${out_defines} ${defines} PARENT_SCOPE)
    set(${out_options} ${options} PARENT_SCOPE)
endfunction()

set(_reflectgen_source_exts
    ".h" ".hh" ".hpp" ".hxx" ".inl"
    ".cpp" ".cc" ".cxx"
)

function(reflectgen_add_unit)
    set(options)
    set(one_value_args TARGET SOURCE_TARGET OUTPUT_IR)
    set(multi_value_args INPUTS INCLUDE_DIRS DEFINES CLANG_ARGS)
    cmake_parse_arguments(RFG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(NOT RFG_TARGET)
        message(FATAL_ERROR "reflectgen_add_unit requires TARGET")
    endif()

    if(NOT RFG_OUTPUT_IR)
        set(RFG_OUTPUT_IR "${CMAKE_CURRENT_BINARY_DIR}/reflectgen/${RFG_TARGET}/${RFG_TARGET}.rf.json")
    endif()

    set(inputs ${RFG_INPUTS})
    if(NOT inputs AND RFG_SOURCE_TARGET)
        _reflectgen_collect_target_sources(inputs "${RFG_SOURCE_TARGET}")
    endif()

    set(unit_dir "${CMAKE_CURRENT_BINARY_DIR}/reflectgen/${RFG_TARGET}")
    set(input_list "${unit_dir}/${RFG_TARGET}.inputs.txt")
    file(MAKE_DIRECTORY "${unit_dir}")
    file(WRITE "${input_list}" "")
    foreach(input IN LISTS inputs)
        file(APPEND "${input_list}" "${input}\n")
    endforeach()

    set(command_args
        -m reflectgen scan
        --target "${RFG_TARGET}"
        --input-list "${input_list}"
        --output-ir "${RFG_OUTPUT_IR}"
    )

    if(RFG_SOURCE_TARGET)
        list(APPEND command_args --source-target "${RFG_SOURCE_TARGET}")
    endif()
    foreach(dir IN LISTS RFG_INCLUDE_DIRS)
        list(APPEND command_args --include-dir "${dir}")
    endforeach()
    foreach(def IN LISTS RFG_DEFINES)
        list(APPEND command_args --define "${def}")
    endforeach()
    foreach(arg IN LISTS RFG_CLANG_ARGS)
        list(APPEND command_args --clang-arg "${arg}")
    endforeach()

    add_custom_command(
        OUTPUT "${RFG_OUTPUT_IR}"
        COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${REFLECTGEN_ROOT}" "${Python3_EXECUTABLE}" ${command_args}
        DEPENDS ${inputs} "${input_list}"
        VERBATIM
        COMMENT "ReflectGen scan ${RFG_TARGET}"
    )

    add_custom_target("${RFG_TARGET}" DEPENDS "${RFG_OUTPUT_IR}")
    set_property(TARGET "${RFG_TARGET}" PROPERTY REFLECTGEN_UNIT_IR "${RFG_OUTPUT_IR}")
endfunction()

function(reflectgen_add_registry)
    set(options TRANSITIVE_SOURCE_TARGETS)
    set(one_value_args TARGET OUTPUT_DIR)
    set(multi_value_args SOURCE_TARGETS UNITS LINK_TO EXTRA_INPUTS EXCLUDE_INPUTS)
    cmake_parse_arguments(RFG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(NOT RFG_TARGET)
        message(FATAL_ERROR "reflectgen_add_registry requires TARGET")
    endif()

    if(NOT RFG_OUTPUT_DIR)
        set(RFG_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/reflectgen/${RFG_TARGET}/generated")
    endif()

    set(units ${RFG_UNITS})
    set(registry_includes)
    set(registry_defines)
    set(registry_options)
    foreach(source_target IN LISTS RFG_SOURCE_TARGETS)
        if(NOT TARGET "${source_target}")
            message(FATAL_ERROR "ReflectGen SOURCE_TARGET does not exist: ${source_target}")
        endif()

        _reflectgen_collect_target_sources(auto_inputs "${source_target}")
        _reflectgen_collect_target_usage(target_includes target_defines target_options "${source_target}")
        list(APPEND registry_includes ${target_includes})
        list(APPEND registry_defines ${target_defines})
        list(APPEND registry_options ${target_options})
        list(APPEND auto_inputs ${RFG_EXTRA_INPUTS})
        foreach(excluded IN LISTS RFG_EXCLUDE_INPUTS)
            list(REMOVE_ITEM auto_inputs "${excluded}")
        endforeach()

        set(unit_target "${RFG_TARGET}_${source_target}_Unit")
        reflectgen_add_unit(
            TARGET "${unit_target}"
            SOURCE_TARGET "${source_target}"
            INPUTS ${auto_inputs}
        )
        list(APPEND units "${unit_target}")
    endforeach()

    if(NOT units)
        message(FATAL_ERROR "reflectgen_add_registry requires SOURCE_TARGETS or UNITS")
    endif()

    set(ir_paths)
    foreach(unit IN LISTS units)
        if(NOT TARGET "${unit}")
            message(FATAL_ERROR "ReflectGen unit target does not exist: ${unit}")
        endif()
        get_target_property(unit_ir "${unit}" REFLECTGEN_UNIT_IR)
        if(NOT unit_ir)
            message(FATAL_ERROR "Target is not a ReflectGen unit: ${unit}")
        endif()
        list(APPEND ir_paths "${unit_ir}")
    endforeach()

    set(registry_dir "${CMAKE_CURRENT_BINARY_DIR}/reflectgen/${RFG_TARGET}")
    set(ir_list "${registry_dir}/${RFG_TARGET}.irs.txt")
    file(MAKE_DIRECTORY "${registry_dir}")
    file(WRITE "${ir_list}" "")
    foreach(ir IN LISTS ir_paths)
        file(APPEND "${ir_list}" "${ir}\n")
    endforeach()

    set(gen_h "${RFG_OUTPUT_DIR}/${RFG_TARGET}.gen.h")
    set(gen_cpp "${RFG_OUTPUT_DIR}/${RFG_TARGET}.gen.cpp")
    set(gen_ir "${RFG_OUTPUT_DIR}/${RFG_TARGET}.registry.json")

    add_custom_command(
        OUTPUT "${gen_h}" "${gen_cpp}" "${gen_ir}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${RFG_OUTPUT_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${REFLECTGEN_ROOT}" "${Python3_EXECUTABLE}" -m reflectgen generate
            --target "${RFG_TARGET}"
            --input-ir-list "${ir_list}"
            --output-dir "${RFG_OUTPUT_DIR}"
        DEPENDS ${units} ${ir_paths} "${ir_list}"
        VERBATIM
        COMMENT "ReflectGen generate ${RFG_TARGET}"
    )

    set_source_files_properties("${gen_cpp}" "${gen_h}" PROPERTIES GENERATED TRUE)
    add_library("${RFG_TARGET}" STATIC "${gen_cpp}" "${gen_h}")
    target_include_directories("${RFG_TARGET}"
        PUBLIC
            "${RFG_OUTPUT_DIR}"
            "${REFLECTGEN_ROOT}/runtime"
        PRIVATE
            ${registry_includes}
    )
    target_compile_definitions("${RFG_TARGET}" PRIVATE ${registry_defines})
    target_compile_options("${RFG_TARGET}" PRIVATE ${registry_options})

    foreach(consumer IN LISTS RFG_LINK_TO)
        if(NOT TARGET "${consumer}")
            message(FATAL_ERROR "ReflectGen LINK_TO target does not exist: ${consumer}")
        endif()
        target_link_libraries("${consumer}" PRIVATE "${RFG_TARGET}")
    endforeach()
endfunction()
